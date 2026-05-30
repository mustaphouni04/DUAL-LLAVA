import torch
from torch import nn
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from dataclasses import dataclass, field
import transformers
import torch.distributed as dist
from llava.train.train import make_supervised_data_module
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token

from typing import Optional, Any
from llava.model.llava_arch import *


class SharedSimpleEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim))

    def forward(self, x):
        z = self.encoder(x)
        return z 


class SimplePostProjectorHead(nn.Module):
    def __init__(self, latent_dim:int, target_dim:int, dtype=torch.float32, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, target_dim),
        )

        if device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, latent):
        out = self.net(latent)

        return out

@dataclass
class ImageFeatures:
    pixel_values: torch.tensor

class SimpleMultiModalProjector(nn.Module):
    def __init__(self, shared_encoder: SharedSimpleEncoder, head: SimplePostProjectorHead, in_dim: int):
        super().__init__()
        self.shared = shared_encoder
        self.head = head

    def forward(self, x: ImageFeatures):
        x = x.to(dtype=torch.bfloat16)
        latent = self.shared(x)
        return self.head(latent)

class SimpleUnifiedModel(nn.Module):
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

        self.log_var_a = nn.Parameter(torch.zeros(1))
        self.log_var_b = nn.Parameter(torch.zeros(1))

        self.modelA.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
        self.modelB.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
        vision_tower = self.modelA.get_vision_tower().to(dtype=torch.bfloat16)
        self.modelB.get_model().vision_tower = vision_tower
        vision_tower.requires_grad_(False)
        self.image_processor = vision_tower.image_processor

        vision_tower_in_dim = vision_tower.config.hidden_size
        self.shared_encoder = SharedSimpleEncoder(in_dim=vision_tower_in_dim, latent_dim=latent_dim)

        outA_dim = self.modelA.config.hidden_size
        outB_dim = self.modelB.config.hidden_size

        dtypeA = self.modelA.get_input_embeddings().weight.dtype
        dtypeB = self.modelB.get_input_embeddings().weight.dtype

        self.headA = SimplePostProjectorHead(latent_dim, outA_dim, dtype=dtypeA)
        self.headB = SimplePostProjectorHead(latent_dim, outB_dim, dtype=dtypeB)

        self.modelA.get_model().mm_projector = SimpleMultiModalProjector(
            self.shared_encoder, self.headA, in_dim=1024
            )

        self.modelB.get_model().mm_projector = SimpleMultiModalProjector(
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

        precision_a = torch.exp(-self.log_var_a)
        precision_b = torch.exp(-self.log_var_b)

        total_loss = precision_a * loss_a + precision_b * loss_b + self.log_var_a + self.log_var_b
        
        #total_loss = loss_a + loss_b

        return total_loss, loss_a.detach(), loss_b.detach()

    def get_trainable_parameters(self):
        return list(self.shared_encoder.parameters()) + \
               list(self.headA.parameters()) + \
               list(self.headB.parameters()) 


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

