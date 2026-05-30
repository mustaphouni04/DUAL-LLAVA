import os
import random
import shutil
import math
from copy import deepcopy
from dataclasses import dataclass, field
import deepspeed
import wandb
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, List
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
import evaluate
from transformers import get_scheduler, HfArgumentParser
from tqdm import tqdm

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import torch.distributed as dist
from llava.train.train import make_supervised_data_module
from models import SimpleUnifiedModel
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from llava.model.llava_arch import *


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
class ImageFeatures:
    pixel_values: torch.tensor

class MultiModalProjector(nn.Module):
    def __init__(self, shared_encoder: SharedEncoder, head: PostProjectorHead, in_dim: int):
        super().__init__()
        self.shared = shared_encoder
        self.head = head

        self.input_adapter = nn.Linear(in_dim, shared_encoder.input_dim)

    def forward(self, x: ImageFeatures):
        x = x.to(dtype=torch.bfloat16)
        latent = self.shared(self.input_adapter(x))
        return self.head(latent)

class UnifiedModel(nn.Module):
    def __init__(self, modelA_ckpt, modelB_ckpt, latent_dim):
        super().__init__()

        print(f"Loading Model A ({modelA_ckpt})...")
        self.modelA = LlavaLlamaForCausalLM.from_pretrained(
            modelA_ckpt, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )


        print(f"Loading Model B ({modelB_ckpt})...")
        self.modelB = LlavaLlamaForCausalLM.from_pretrained(
            modelB_ckpt, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )

        self.modelA.config.use_cache = False
        self.modelB.config.use_cache = False
        self.modelA.model.requires_grad_(False)
        self.modelB.model.requires_grad_(False)

        self.modelA.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
        self.modelB.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
        vision_tower = self.modelA.get_vision_tower().to(dtype=torch.bfloat16)
        self.modelB.get_model().vision_tower = vision_tower
        vision_tower.requires_grad_(False)
        self.image_processor = vision_tower.image_processor

        vision_tower_in_dim = vision_tower.config.hidden_size
        self.shared_encoder = SharedEncoder(in_dim=vision_tower_in_dim, latent_dim=latent_dim, init_scale=0.05)

        outA_dim = self.modelA.config.hidden_size
        outB_dim = self.modelB.config.hidden_size

        dtypeA = self.modelA.get_input_embeddings().weight.dtype
        dtypeB = self.modelB.get_input_embeddings().weight.dtype

        self.headA = PostProjectorHead(latent_dim, outA_dim, use_layernorm=True, init_scale=0.05, dtype=dtypeA)
        self.headB = PostProjectorHead(latent_dim, outB_dim, use_layernorm=True, init_scale=0.05, dtype=dtypeB)

        self.modelA.get_model().mm_projector = MultiModalProjector(
            self.shared_encoder, self.headA, in_dim=1024
            )

        self.modelB.get_model().mm_projector = MultiModalProjector(
            self.shared_encoder, self.headB, in_dim=1024
            )

    def forward(self, batch_a, batch_b):
        device = next(self.parameters()).device

        images_a = batch_a.pop("images").to(device, non_blocking=True)
        input_ids_a = batch_a["input_ids"].to(device, non_blocking=True)
        labels_a = batch_a["labels"].to(device, non_blocking=True)
        attn_mask_a = batch_a["attention_mask"].to(device, non_blocking=True)

        outputs_a = self.modelA(
            input_ids=input_ids_a,
            attention_mask=attn_mask_a,
            images=images_a,
            labels=labels_a)

        loss_a = outputs_a.loss

        images_b = batch_b.pop("images").to(device, non_blocking=True)
        input_ids_b = batch_b["input_ids"].to(device, non_blocking=True)
        labels_b = batch_b["labels"].to(device, non_blocking=True)
        attn_mask_b = batch_b["attention_mask"].to(device, non_blocking=True)


        outputs_b = self.modelB(
            input_ids=input_ids_b,
            attention_mask=attn_mask_b,
            images=images_b,
            labels=labels_b)

        loss_b = outputs_b.loss

        #with torch.no_grad():
        #    scale_a = (loss_b.detach().mean() / (loss_a.detach().mean() + 1e-8)).clamp(0.1, 10)
        
        total_loss = loss_a + loss_b

        return total_loss, loss_a.detach(), loss_b.detach()

    def get_trainable_parameters(self):
        return list(self.shared_encoder.parameters()) + \
               list(self.headA.parameters()) + \
               list(self.headB.parameters()) + \
               list(self.modelA.get_model().mm_projector.input_adapter.parameters()) + \
               list(self.modelB.get_model().mm_projector.input_adapter.parameters())

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
    model_max_length: int = 1050

def save_checkpoint(model_engine, save_dir):
    torch.distributed.barrier()
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(save_dir):
        if file.startswith("global"):
            shutil.rmtree(os.path.join(save_dir, file), ignore_errors=True)

    model_engine.save_checkpoint(save_dir)
    print(f"Checkpoint saved to {save_dir}")


def decode_after_image(tokenizer, input_ids_batch, image_token_id=-200):
    decoded_texts = []
    for ids in input_ids_batch:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if image_token_id in ids:
            start_idx = ids.index(image_token_id) + 1  
            ids = ids[start_idx:]
        else:
            ids = ids
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[0].strip()
        decoded_texts.append(text)
    return decoded_texts

def make_rouge_metrics():
    rouge_A = evaluate.load("rouge")
    rouge_B = evaluate.load("rouge")
    return rouge_A, rouge_B

def evaluate_models(
    model_engine,
    batchA,
    batchB,
    images_a,
    images_b,
    tokenizer_A,
    tokenizer_B,
    rouge_A=None,
    rouge_B=None,
    sample_n: int = 2,                 
    max_new_tokens: int = 32,          
    seed: Optional[int] = 1234,       
):

    model_engine.eval()
    device = model_engine.device

    for tokenizer in [tokenizer_A, tokenizer_B]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    created_local_rouge = False
    if rouge_A is None or rouge_B is None:
        rouge_A_local, rouge_B_local = make_rouge_metrics()
        if rouge_A is None:
            rouge_A = rouge_A_local
        if rouge_B is None:
            rouge_B = rouge_B_local
        created_local_rouge = True

    if batchA is None or batchB is None:
        model_engine.train()
        return None, None

    batch_size_a = batchA["input_ids"].shape[0]
    batch_size_b = batchB["input_ids"].shape[0]
    rng = random.Random(seed)
    idxs_a = list(range(batch_size_a))
    idxs_b = list(range(batch_size_b))
    if batch_size_a > sample_n:
        idxs_a = rng.sample(idxs_a, sample_n)
    if batch_size_b > sample_n:
        idxs_b = rng.sample(idxs_b, sample_n)

    def prepare_subset(tokenizer, batch, idxs, image_token_index, images, conv_template):
        labels_full = batch["labels"][idxs].to(device, non_blocking=True)  # (k, seq_len)
        prompts = decode_after_image(tokenizer, batch["input_ids"][idxs])
        prompt_texts = []
        for p in prompts:
            conv = conv_templates["plain"].copy()
            processed = DEFAULT_IMAGE_TOKEN + "\n" + p
            conv.append_message(conv.roles[0], processed)
            conv.append_message(conv.roles[1], None)
            prompt_texts.append(conv.get_prompt())

        input_ids_list = [
            tokenizer_image_token(t, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0)
            for t in prompt_texts
        ]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).to(device)
        images_subset = images[idxs].to(device, non_blocking=True)

        return labels_full, input_ids_padded, attention_mask, images_subset

    with torch.no_grad():
        labels_a, input_ids_padded_a, attention_mask_a, images_a_sub = prepare_subset(
            tokenizer_A, batchA, idxs_a, IMAGE_TOKEN_INDEX, images_a, conv_templates
        )

        outputs_ids_a = model_engine.module.modelA.generate(
            inputs=input_ids_padded_a,
            images=images_a_sub,
            attention_mask=attention_mask_a,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer_A.pad_token_id,
            eos_token_id=tokenizer_A.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        batch_pred_a = tokenizer_A.batch_decode(outputs_ids_a, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pad_id_A = tokenizer_A.pad_token_id
        labels_a_for_decode = labels_a.clone()
        labels_a_for_decode[labels_a_for_decode == -100] = pad_id_A

        labels_a_list = [lab.cpu().tolist() for lab in labels_a_for_decode]
        batch_lab_a = tokenizer_A.batch_decode(labels_a_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        labels_b, input_ids_padded_b, attention_mask_b, images_b_sub = prepare_subset(
            tokenizer_B, batchB, idxs_b, IMAGE_TOKEN_INDEX, images_b, conv_templates
        )

        outputs_ids_b = model_engine.module.modelB.generate(
            inputs=input_ids_padded_b,
            images=images_b_sub,
            attention_mask=attention_mask_b,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer_B.pad_token_id,
            eos_token_id=tokenizer_B.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        batch_pred_b = tokenizer_B.batch_decode(outputs_ids_b, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pad_id_B = tokenizer_B.pad_token_id
        labels_b_for_decode = labels_b.clone()
        labels_b_for_decode[labels_b_for_decode == -100] = pad_id_B
        labels_b_list = [lab.cpu().tolist() for lab in labels_b_for_decode]
        batch_lab_b = tokenizer_B.batch_decode(labels_b_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        result_A = None
        result_B = None
        
        rouge_metric = evaluate.load("rouge")
        result_A = rouge_metric.compute(predictions=batch_pred_a, references=batch_lab_a) 
        result_B = rouge_metric.compute(predictions=batch_pred_b, references=batch_lab_b)

    model_engine.train()
    return result_A, result_B

def train_shared_pca(
    modelA_ckpt: str,
    modelB_ckpt: str,
    latent_dim=2048,
    per_device_train_batch_size=9,
    epochs=4,
    lr=2e-3,
    shared_lr_scale=0.1,
    weight_decay=0.0,
    save_dir="./out_shared_pca",
    load_dir:Optional[str] = None,
    eval_steps=100
    ):

    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    rouge_A, rouge_B = make_rouge_metrics()

    if training_args.local_rank == 0:
        run = wandb.init(
            project="SharedPCA",
            notes="Qwen2.5+Gemma2",
            config={"Model_A": modelA_ckpt,
                    "Model_B": modelB_ckpt,
                    "latent_dim": latent_dim,
                    "per_device_batch_size": per_device_train_batch_size,
                    "num_epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "save_dir": save_dir})
    
    deepspeed.init_distributed()
    torch.cuda.set_device(training_args.local_rank)

    model = UnifiedModel(modelA_ckpt, modelB_ckpt, latent_dim)

    tokenizer_A = transformers.AutoTokenizer.from_pretrained(modelA_ckpt, use_fast=False)
    tokenizer_B = transformers.AutoTokenizer.from_pretrained(modelB_ckpt, use_fast=False)

    data_args_A = DataArguments(image_processor=model.image_processor)
    data_args_B = DataArguments(image_processor=model.image_processor)

    data_module_A = make_supervised_data_module(tokenizer_A, data_args_A)
    data_module_B = make_supervised_data_module(tokenizer_B, data_args_B)

    samplerA = torch.utils.data.distributed.DistributedSampler(data_module_A["train_dataset"], shuffle=True)
    samplerB = torch.utils.data.distributed.DistributedSampler(data_module_B["train_dataset"], shuffle=True)

    loaderA = DataLoader(
        data_module_A["train_dataset"],
        batch_size=per_device_train_batch_size,
        sampler=samplerA,
        shuffle=False,
        collate_fn=data_module_A["data_collator"],
    )
    loaderB = DataLoader(
        data_module_B["train_dataset"],
        batch_size=per_device_train_batch_size,
        sampler=samplerB,
        shuffle=False,
        collate_fn=data_module_B["data_collator"],
    )

    num_batches = min(len(loaderA), len(loaderB))
    total_steps = epochs * num_batches
    warmup_steps = int(total_steps * 0.03)

    print(f"[Rank {training_args.local_rank}] Setting up differential LRs...")
    shared_lr = lr * shared_lr_scale 
    head_lr = lr                     

    shared_params = list(model.shared_encoder.parameters()) + \ 
                    list(model.modelA.get_model().mm_projector.input_adapter.parameters()) + \
                    list(model.modelB.get_model().mm_projector.input_adapter.parameters())

    head_params = list(model.headA.parameters()) + list(model.headB.parameters())

    optimizer_grouped_parameters = [
        {"params": shared_params, "lr": shared_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=training_args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=model.get_trainable_parameters(),
        config="ds_config.json")

    if load_dir is not None:
        print(f"[Rank {training_args.local_rank}] Loading checkpoint from: {load_dir}")
        model_engine.load_checkpoint(load_dir)
        print(f"[Rank {training_args.local_rank}] Checkpoint loaded successfully.")

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.in_features))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    #print(f"[Rank {training_args.local_rank}] Re-initializing weights for headB...")
    #model_engine.module.headB.apply(weights_init)
    eval_mode = False
    
    model_engine.train()

    for epoch in range(epochs):
        samplerA.set_epoch(epoch)
        samplerB.set_epoch(epoch)

        iterA = iter(loaderA)
        iterB = iter(loaderB)

        num_batches = min(len(loaderA), len(loaderB))
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", disable=(training_args.local_rank != 0))
        
        eval_counter = 0

        for step in progress_bar:
            batch_a = next(iterA)
            batch_b = next(iterB)

            images_a = batch_a["images"].to(next(model_engine.parameters()).device)
            images_b = batch_b["images"].to(next(model_engine.parameters()).device)

            total_loss, loss_a, loss_b = model_engine(batch_a, batch_b)

            model_engine.backward(total_loss)
            model_engine.step() 

            if training_args.local_rank == 0:
                with torch.no_grad():
                    modelA = model_engine.module.modelA.get_model()
                    vision_tower = modelA.get_vision_tower()
                    mm_projector = modelA.mm_projector

                    image_features = vision_tower(images_a.to(dtype=torch.bfloat16))

                    adapted = mm_projector.input_adapter(image_features)
                    latent = mm_projector.shared(adapted) 

                    scale_mean = latent.mean().item()
                    scale_std = latent.std().item()

                    wandb.log({
                        "encoder_output_mean": scale_mean,
                        "encoder_output_std": scale_std,
                    }) 

            if training_args.local_rank == 0:
                current_lr = model_engine.get_lr()[0]
                grad_norm = model_engine.get_global_grad_norm()

                wandb.log({
                    "loss_a": loss_a.item(),
                    "loss_b": loss_b.item(),
                    "total_loss": total_loss.item(),
                    "grad_norm": grad_norm,
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    "step": step
                })
            eval_counter += 1

            if eval_counter % eval_steps == 0 and step != 0 and training_args.local_rank == 0:
                rouge_A, rouge_B = evaluate_models(
                    model_engine,  
                    batch_a,
                    batch_b,
                    images_a,
                    images_b,
                    tokenizer_A,
                    tokenizer_B,
                    rouge_A,
                    rouge_B
                    )

                wandb.log({
                        "rouge_A": rouge_A,
                        "rouge_B": rouge_B,
                        })

            if step % 2000 == 0 and step!=0:
                save_checkpoint(model_engine, save_dir)

    print(f"[Rank {training_args.local_rank}] Reached end of script, waiting at barrier.")
    save_checkpoint(model_engine, save_dir)
    torch.distributed.barrier()
    

if __name__ == "__main__":
    train_shared_pca(
        modelA_ckpt="Qwen/Qwen2.5-1.5B-Instruct",
        modelB_ckpt="google/gemma-2-2b-it"
    )
