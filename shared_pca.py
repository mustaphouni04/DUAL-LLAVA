import os
import math
from dataclasses import dataclass, field
import wandb
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, List
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import get_scheduler
from tqdm import tqdm

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import make_supervised_data_module
from llava.model.llava_arch import *

_PER_MODEL_POST_HEADS = {}
_ACTIVE_SHARED_PROJECTOR_IDX = None

@contextmanager
def using_projector_index(idx: int):
    global _ACTIVE_SHARED_PROJECTOR_IDX
    old = _ACTIVE_SHARED_PROJECTOR_IDX
    _ACTIVE_SHARED_PROJECTOR_IDX = idx
    try:
        yield
    finally:
        _ACTIVE_SHARED_PROJECTOR_IDX = old

def current_projector_index():
    return _ACTIVE_SHARED_PROJECTOR_IDX

class SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, use_layernorm=True, init_scale: float = 0.1):
        super().__init__()
        self.input_dim = in_dim
        self.encoder = nn.Linear(in_dim, latent_dim, bias=False)
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.ln = nn.LayerNorm(latent_dim, eps=1e-6) if use_layernorm else None
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        std_e = 1.0 / math.sqrt(in_dim)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=std_e)
        for l in self.bottleneck:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, mean=0.0, std=1.0 / math.sqrt(l.in_features))
                if l.bias is not None:
                    nn.init.zeros_(l.bias)

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        if self.ln is not None:
            z = self.ln(z)
        return z * self.scale

class SharedEncoderWrapper(nn.Module):
    def __init__(self, shared_encoder: SharedEncoder, model_index: int, in_dim: int):
        super().__init__()
        self.shared = shared_encoder
        self.model_index = model_index
        self.input_adapter = nn.Linear(in_dim, shared_encoder.input_dim)

    def forward(self, x):
        target_device = next(self.shared.parameters()).device
        x = x.to(target_device)
        latent = self.shared(self.input_adapter(x))
        active_idx = current_projector_index()
        if active_idx is None:
            raise RuntimeError(
                "Projector index not set. You must wrap the model call in 'with using_projector_index(idx):'"
            )

        head = _PER_MODEL_POST_HEADS[active_idx]
        target_head_device = next(head.parameters()).device
        final_features = head(latent.to(target_head_device))

        return final_features

class PostProjectorHead(nn.Module):
    def __init__(self, latent_dim:int, target_dim:int, use_layernorm:bool=True, init_scale:float=0.1, dtype=torch.float32, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, target_dim),
        )
        self.ln = nn.LayerNorm(target_dim, eps=1e-6) if use_layernorm else None
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=dtype, device=device))
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, mean=0.0, std=1.0 / math.sqrt(l.in_features))
                if l.bias is not None:
                    nn.init.zeros_(l.bias)
        if self.ln is not None:
            pass

        if device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, latent):
        out = self.net(latent)
        if self.ln is not None:
            out = self.ln(out)
        out = out * self.scale.to(dtype=out.dtype, device=out.device)
        return out
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="plain")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14")
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: Optional[str] = "../cc3m_filtered/blip_laion_cc_sbu_558k.json"
    image_folder: Optional[str] = "../cc3m_filtered/images"
    image_aspect_ratio: str = 'square'
    is_multimodal: bool = True
    lazy_preprocess: bool = False
    image_processor: Optional[Any] = None
    mm_use_im_start_end: bool = field(default=False)
    version: Optional[str] = field(default="plain")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14")
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    model_max_length: int = 1000

def train_shared_pca(
    modelA_ckpt: str,
    modelB_ckpt: str,
    latent_dim=2048,
    per_device_train_batch_size=8,
    epochs=1,
    lr=3e-3,
    weight_decay=0.0,
    save_dir="./out_shared_pca"
):
    run = wandb.init(
        project="SharedPCA",
        notes="Qwen2.5+Gemma2 with PCGrad",
        config={"Model_A": modelA_ckpt,
                "Model_B": modelB_ckpt,
                "latent_dim": latent_dim,
                "per_device_batch_size": per_device_train_batch_size,
                "num_epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "save_dir": save_dir})
                
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least two GPUs.")

    deviceA = torch.device("cuda:0")
    deviceB = torch.device("cuda:1")
    shared_device = torch.device("cuda:2")

    training_args, data_args_A, data_args_B = TrainingArguments(), DataArguments(), DataArguments()

    print(f"Loading Model A ({modelA_ckpt}) onto {deviceA}...")
    modelA = LlavaLlamaForCausalLM.from_pretrained(
        modelA_ckpt, attn_implementation="eager", torch_dtype=torch.bfloat16
    ).to(deviceA)
    print(f"Loading Model B ({modelB_ckpt}) onto {deviceB}...")
    modelB = LlavaLlamaForCausalLM.from_pretrained(
        modelB_ckpt, attn_implementation="eager", torch_dtype=torch.bfloat16
    ).to(deviceB)

    modelA.config.use_cache = False
    modelB.config.use_cache = False
    modelA.model.requires_grad_(False)
    modelB.model.requires_grad_(False)

    tokenizer_A = transformers.AutoTokenizer.from_pretrained(modelA_ckpt, use_fast=False)
    tokenizer_B = transformers.AutoTokenizer.from_pretrained(modelB_ckpt, use_fast=False)

    for tokenizer in [tokenizer_A, tokenizer_B]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    modelA.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
    modelB.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
    vision_tower = modelA.get_vision_tower().to(dtype=torch.bfloat16).to(shared_device)
    modelB.get_model().vision_tower = vision_tower
    
    data_args_A.image_processor = vision_tower.image_processor
    data_args_B.image_processor = vision_tower.image_processor
    vision_tower.requires_grad_(False)

    in_dim = vision_tower.config.hidden_size
    shared_encoder = SharedEncoder(in_dim=in_dim, latent_dim=latent_dim, init_scale=0.05).to(shared_device)

    outA = modelA.config.hidden_size
    outB = modelB.config.hidden_size

    embed_A = modelA.get_input_embeddings().weight
    embed_B = modelB.get_input_embeddings().weight
    dtypeA = embed_A.dtype
    dtypeB = embed_B.dtype
    devA = embed_A.device
    devB = embed_B.device

    headA = PostProjectorHead(latent_dim, outA, use_layernorm=True, init_scale=0.05, dtype=dtypeA, device=devA)
    headB = PostProjectorHead(latent_dim, outB, use_layernorm=True, init_scale=0.05, dtype=dtypeB, device=devB)

    _PER_MODEL_POST_HEADS[0] = headA
    _PER_MODEL_POST_HEADS[1] = headB

    modelA.get_model().mm_projector = SharedEncoderWrapper(shared_encoder, 0, in_dim=1024).to(shared_device)
    modelB.get_model().mm_projector = SharedEncoderWrapper(shared_encoder, 1, in_dim=1024).to(shared_device)

    data_module_A = make_supervised_data_module(tokenizer_A, data_args_A)
    data_module_B = make_supervised_data_module(tokenizer_B, data_args_B)

    loaderA = DataLoader(
        data_module_A["train_dataset"],
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_module_A["data_collator"],
    )
    loaderB = DataLoader(
        data_module_B["train_dataset"],
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_module_B["data_collator"],
    )

    optimizer = torch.optim.AdamW(
        list(shared_encoder.parameters()) + list(headA.parameters()) + list(headB.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    total_steps = epochs * min(len(loaderA), len(loaderB))
    warmup_steps = int(total_steps * 0.03)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    modelA.eval()
    modelB.eval()

    for epoch in range(epochs):
        iterA = iter(loaderA)
        iterB = iter(loaderB)
        
        num_batches = min(len(loaderA), len(loaderB))
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")

        for step in progress_bar:
            try:
                batch_a = next(iterA)
                batch_b = next(iterB)
            except StopIteration:
                break

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images_a = batch_a.pop("images").to(shared_device, non_blocking=True)
                input_ids_a = batch_a["input_ids"].to(deviceA)
                labels_a = batch_a["labels"].to(deviceA)
                attn_mask_a = batch_a["attention_mask"].to(deviceA)
                with using_projector_index(0):
                    outputs_a = modelA(input_ids=input_ids_a, attention_mask=attn_mask_a, images=images_a, labels=labels_a)
                loss_a = outputs_a.loss

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images_b = batch_b.pop("images").to(shared_device, non_blocking=True)
                input_ids_b = batch_b["input_ids"].to(deviceB)
                labels_b = batch_b["labels"].to(deviceB)
                attn_mask_b = batch_b["attention_mask"].to(deviceB)
                with using_projector_index(1):
                    outputs_b = modelB(input_ids=input_ids_b, attention_mask=attn_mask_b, images=images_b, labels=labels_b)
                loss_b = outputs_b.loss

            total_loss = loss_a.to(shared_device) + loss_b.to(shared_device)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(shared_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(headA.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(headB.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() 
            
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "loss_a": loss_a.item(),
                "loss_b": loss_b.item(),
                "total_loss": total_loss.item(),
                "learning_rate": current_lr,
                "epoch": epoch,
                "step": step
            })
            
            progress_bar.set_postfix(loss_A=f"{loss_a.item():.4f}", loss_B=f"{loss_b.item():.4f}", lr=f"{current_lr:.2e}")

        
            if step % 6000 == 0 and step != 0:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(shared_encoder.state_dict(), save_dir / "shared_encoder.pt")
                torch.save(headA.state_dict(), save_dir / "headA.pt")
                torch.save(headB.state_dict(), save_dir / "headB.pt")
                print(f"Checkpoints saved in {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    train_shared_pca(
        modelA_ckpt="Qwen/Qwen2.5-1.5B-Instruct",
        modelB_ckpt="google/gemma-2-2b-it"
    )
