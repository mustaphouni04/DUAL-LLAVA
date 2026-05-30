import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from PIL import Image
import requests
from io import BytesIO
from cog import BasePredictor, Input, Path, ConcatenateIterator
from threading import Thread
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Any

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

class MultiModalProjector(nn.Module):
    def __init__(self, shared_encoder: SharedEncoder, head: PostProjectorHead, in_dim: int):
        super().__init__()
        self.shared = shared_encoder
        self.head = head
        self.input_adapter = nn.Linear(in_dim, shared_encoder.input_dim)
        
    def forward(self, x: torch.Tensor):
        
        latent = self.input_adapter(x.to(self.input_adapter.weight.dtype))
        
        latent = self.shared(latent)
        
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
        
        self.modelA.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
        self.modelB.get_model().initialize_vision_modules(model_args=ModelArguments(), fsdp=None)
        vision_tower = self.modelA.get_vision_tower().to(dtype=torch.bfloat16)
        self.modelB.get_model().vision_tower = vision_tower
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

def load_image(image_file):
    if image_file.startswith('http'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class Predictor(BasePredictor):
    def setup(self) -> None:
        disable_torch_init()

        model_a_ckpt = "Qwen/Qwen2.5-1.5B-Instruct"
        model_b_ckpt = "google/gemma-2-2b-it"
        latent_dim = 2048 
        checkpoint_dir = "../out_shared_pca"
        
        print("Loading and configuring tokenizers...")
        self.tokenizers = {
            'qwen': AutoTokenizer.from_pretrained(model_a_ckpt, use_fast=False),
            'gemma': AutoTokenizer.from_pretrained(model_b_ckpt, use_fast=False)
        }

        print("Instantiating full UnifiedModel architecture...")
        self.unified_model = UnifiedModel(model_a_ckpt, model_b_ckpt, latent_dim)

        try:
            with open(os.path.join(checkpoint_dir, 'latest'), 'r') as f:
                tag = f.read().strip()
            print(f"Found latest checkpoint tag: {tag}")
        except FileNotFoundError:
            print(f"Error: Could not find 'latest' file in {checkpoint_dir}.")
            print("Please ensure the checkpoint directory is correct.")
            raise

        ckpt_path = os.path.join(checkpoint_dir, tag, 'mp_rank_00_model_states.pt')
        print(f"Loading weights from: {ckpt_path}")

        try:
            full_state_dict = torch.load(ckpt_path, map_location='cpu')
            model_state_dict = full_state_dict['module']
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {ckpt_path}")
            print("This can happen if you are using ZeRO-3 or a different rank setup.")
            raise
        except KeyError:
            print(f"Error: Could not find 'module' key in the checkpoint file.")
            raise

        self.unified_model.load_state_dict(model_state_dict)

        print("Moving model to GPU and setting to eval mode...")
        self.unified_model.to("cuda").eval().to(torch.bfloat16)

        self.models = {
            'qwen': self.unified_model.modelA,
            'gemma': self.unified_model.modelB
        }

        self.image_processor = self.unified_model.image_processor

        print("Setup complete.")


    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        model_choice: str = Input(
            description="Choose which model to use for inference",
            choices=["qwen", "gemma"],
            default="qwen",
        ),
        top_p: float = Input(description="Top-p sampling parameter", default=1.0),
        temperature: float = Input(description="Temperature for sampling", default=0.2),
        max_tokens: int = Input(description="Maximum number of new tokens to generate", default=1024),
    ) -> ConcatenateIterator[str]:

        model = self.models[model_choice]
        tokenizer = self.tokenizers[model_choice]

        conv_mode = "plain" 
        conv = conv_templates[conv_mode].copy()
        image_data = load_image(str(image))
        
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.bfloat16)

        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=20.0)

        with torch.inference_mode():
            thread = Thread(target=model.generate, kwargs=dict(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                streamer=streamer,
                use_cache=True
            ))
            thread.start()

            for new_text in streamer:
                yield new_text
            thread.join()
