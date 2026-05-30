import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from dataclasses import dataclass
import torch.nn as nn
from typing import List, Dict
import os
import requests
from types import SimpleNamespace
from cog import BasePredictor, Input, ConcatenateIterator
import numpy as np
from tqdm import tqdm
import copy
from scipy.stats import gaussian_kde
import sys
import matplotlib.pyplot as plt
import toml

sys.path.append('../vec2vec')

from utils.utils import load_n_translator
from utils.npz_dataset import NPZDataset

from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

from transformers import AutoTokenizer, GemmaForCausalLM, LlamaForCausalLM, GenerationConfig, Qwen2ForCausalLM, AutoModelForCausalLM

device_map = device = "cuda"

def conversation_formatting(conv: Path | str = "../cc3m_filtered/flickr_conversations.json"):
    path = Path(conv)

    with open(path, "r") as js:
        file = json.load(js)
    
    return file

class FlickrDataset(Dataset):
    def __init__(self):
        self.file = conversation_formatting()
    def __len__(self):
        return len(self.file)
    def __getitem__(self, id):
        image, caption = "../cc3m_filtered" / Path(self.file[id]["image"]), self.file[id]["conversations"][1]["value"]
        image = cv2.imread(image)

        return {"image": image, "caption": caption}

@dataclass
class DataSample:
    image: np.ndarray
    caption: str

class StatAnalysis:
    def __init__(self, ds):
        self.ds = ds
        llava_path = "scripts/checkpoints/llava-Qwen2.5-1.5B-Instruct-pretrain/checkpoint-34883"
        
        self.tokenizer, self.llava_model, self.image_processor, _ = load_pretrained_model(
            llava_path, None, "llava-qwen", device_map=device_map)

        donor_config = self.llava_model.config

        manual_config = copy.deepcopy(donor_config)
        manual_config.architectures = ['Qwen2ForCausalLM']#"LlamaForCausalLM"] 

        manual_config.model_type = "llama"

        self.manual_model = Qwen2ForCausalLM(manual_config).to(dtype=torch.float16).cuda()

        llava_state = self.llava_model.state_dict()
        manual_state = self.manual_model.state_dict()
        
        filtered_state = {}
        for k, v in llava_state.items():
            if "vision_tower" in k or "mm_projector" in k or "image_newline" in k:
                continue
            
            if k in manual_state:
                filtered_state[k] = v
            else:
                pass

        missing, unexpected = self.manual_model.load_state_dict(filtered_state, strict=False)
        
        self.manual_model.generation_config = GenerationConfig.from_model_config(manual_config)
        self.manual_model.generation_config.pad_token_id = self.tokenizer.eos_token_id
    
    def analysis(self):
        pbar = tqdm(ds, desc="Processing Flickr images...")
        moms = 0
        pmax = 0
        for j, sample_dict in enumerate(pbar):
            sample = DataSample(**sample_dict)
            image_tensor = self.image_processor.preprocess(sample.image, return_tensors='pt')['pixel_values'].half().cuda()
            with torch.inference_mode():
                wrapper = self.llava_model.get_model().get_vision_tower()
                hf_vision_model = wrapper.vision_tower 
                image_forward_out = hf_vision_model(image_tensor, output_hidden_states=True)
                
                selected_layer = image_forward_out.hidden_states[-2]
                features_no_cls = selected_layer[:, 1:] 
                visual_embeddings = self.llava_model.get_model().mm_projector(features_no_cls)

                text_str = "\n" + "Describe this image in detail." # + sample.caption
                text_inputs = self.tokenizer(text_str, return_tensors='pt', add_special_tokens=False).to(device_map)
                text_embeddings = self.manual_model.get_input_embeddings()(text_inputs.input_ids)

                visual_embeddings = visual_embeddings.to(dtype=text_embeddings.dtype)
                inputs_embeds_manual = torch.cat([visual_embeddings, text_embeddings], dim=1)
                attention_mask_manual = torch.ones(inputs_embeds_manual.shape[:2], dtype=torch.long, device=device_map)

                outputs = self.manual_model(
                    inputs_embeds=inputs_embeds_manual,
                    attention_mask=attention_mask_manual,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    )

                # 28 attention layers
                attn = outputs.attentions[-1]

                V = visual_embeddings.size(1)
                T = text_embeddings.size(1)

                vision_idx = slice(0, V)
                text_idx = slice(V, V + T)

                vision_to_text = attn[:, :, text_idx, vision_idx]

                mean_heads = vision_to_text.mean(dim=1)

                top_k = mean_heads.topk(k=3, dim=-1).values
                alignment_score = top_k.mean()
                print(alignment_score)

                moms += alignment_score.item()

                if j == 4e3:
                    break

        final_metric = moms / 4e3 #len(pbar)
        print("Top-K vision attention metric: ", final_metric)

class StatAnalysisTarget:
    def __init__(self, ds):
        self.ds = ds
        llava_path = "scripts/checkpoints/llava-Qwen2.5-1.5B-Instruct-pretrain/checkpoint-34883"
        
        self.donor_tokenizer, self.donor_model, self.image_processor, _ = load_pretrained_model(
            llava_path, None, "llava-qwen", device_map=device_map
        )

        gemma_path = "google/gemma-2-2b-it"
        
        self.host_tokenizer = AutoTokenizer.from_pretrained(gemma_path)
        self.host_model = AutoModelForCausalLM.from_pretrained(
            gemma_path, 
            torch_dtype=torch.float32, 
            device_map=device_map
        )
        self.host_model.eval()

        donor_dim = self.donor_model.config.hidden_size # 1536
        host_dim = self.host_model.config.hidden_size   # 2304
        
        cfg_dict = toml.load("../vec2vec/finetuning_unsupervised/unsupervised/config.toml")

        self.cfg = SimpleNamespace(**cfg_dict)

        model_path = os.path.join("../vec2vec/finetuning_unsupervised/unsupervised", 'model.pt')

        encoder_dims = {
            'sup_model': host_dim, 
            'unsup_model': donor_dim 
        }

        translator = load_n_translator(self.cfg, encoder_dims)
        translator.load_state_dict(torch.load(model_path, map_location=device))
        translator.to("cuda")
        translator.to(torch.float32)
        translator.eval()
        self.translator = translator

        print("Setup Complete.")

    def analysis(self):
        torch.manual_seed(42)

        pbar = tqdm(ds, desc="Processing Flickr images...")
        means = []
        moms = 0

        with torch.inference_mode():
            for j, sample in enumerate(pbar):
                sample = DataSample(**sample)

                image_tensor = self.image_processor.preprocess(sample.image, return_tensors='pt')['pixel_values'].cuda()
                vision_tower = self.donor_model.get_model().get_vision_tower()
                image_forward = vision_tower.vision_tower(image_tensor, output_hidden_states=True)
                
                features = image_forward.hidden_states[-2][:, 1:] 
                
                visual_embeds_qwen = self.donor_model.get_model().mm_projector(features)

                mean_embeds = visual_embeds_qwen.mean(dim=-1, keepdim=False).squeeze(0).detach().cpu().numpy()
                means.append(mean_embeds)

                model_input = visual_embeds_qwen.squeeze(0).to(torch.float32)

                ins = {'unsup_model': model_input}
                out_set = {'sup_model'}
                _, visual_embeds_gemma = self.translator(ins=ins, out_set=out_set)
                visual_embeds_gemma = visual_embeds_gemma["sup_model"]["unsup_model"]
                visual_embeds_gemma = visual_embeds_gemma.unsqueeze(0) 
                
                text_str = "\n" + "Describe this image in detail."
                
                text_inputs = self.host_tokenizer(text_str, return_tensors='pt', add_special_tokens=False).to(device_map)
                
                text_embeddings = self.host_model.get_input_embeddings()(text_inputs.input_ids)
                
                visual_embeds_gemma = visual_embeds_gemma.to(dtype=text_embeddings.dtype)
                
                inputs_embeds_manual = torch.cat([visual_embeds_gemma, text_embeddings], dim=1)
                
                attention_mask_manual = torch.ones(inputs_embeds_manual.shape[:2], dtype=torch.long, device=device_map)
                
                outputs = self.host_model(
                    inputs_embeds=inputs_embeds_manual,
                    attention_mask=attention_mask_manual,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    )

                attn = outputs.attentions[-1]

                V = visual_embeds_gemma.size(1)
                T = text_embeddings.size(1)

                vision_idx = slice(0, V)
                text_idx = slice(V, V + T)

                vision_to_text = attn[:, :, text_idx, vision_idx]

                mean_heads = vision_to_text.mean(dim=1)
                  
                top_k = mean_heads.topk(k=3, dim=-1).values
                alignment_score = top_k.mean()

                moms += alignment_score.item()

                if j == 4e3:
                    break

        final_metric = moms / 4e3 #len(pbar)
        print("Top-3 vision attention metric: ", final_metric)

        means = np.stack(means, axis=0).reshape(-1)
        print(means.shape)
        plt.hist(means, bins=50, density=True, alpha=0.4)

        kde = gaussian_kde(means)
        xs = np.linspace(means.min(), means.max(), 500)

        plt.title("Mean embedding distribution (LLaVA Qwen)")
        plt.plot(xs, kde(xs), linewidth=2)
        plt.savefig("histogram.png")


ds = FlickrDataset()

stat = StatAnalysis(ds)
stat.analysis()

        

