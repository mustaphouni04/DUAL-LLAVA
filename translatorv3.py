import torch
import torch.nn as nn
import os
import requests
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from cog import BasePredictor, Input, ConcatenateIterator
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import toml

sys.path.append('../vec2vec')

from utils.utils import load_n_translator
from utils.npz_dataset import NPZDataset

from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

from transformers import AutoTokenizer, GemmaForCausalLM, AutoModelForCausalLM

os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"
device_map = device = "cuda"  

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class Predictor(BasePredictor):
    def setup(self) -> None:
        llava_path = "scripts/checkpoints/llava-Qwen2.5-1.5B-Instruct-pretrain/checkpoint-34883"
        print(f"Loading Donor from {llava_path}...")
        
        self.donor_tokenizer, self.donor_model, self.image_processor, _ = load_pretrained_model(
            llava_path, None, "llava-qwen", device_map=device_map
        )

        gemma_path = "google/gemma-2-2b-it"
        print(f"Loading Host LLM (Native): {gemma_path}...")
        
        self.host_tokenizer = AutoTokenizer.from_pretrained(gemma_path)
        self.host_model = AutoModelForCausalLM.from_pretrained(
            gemma_path, 
            torch_dtype=torch.float32, 
            device_map=device_map
        )
        self.host_model.eval()

        donor_dim = self.donor_model.config.hidden_size # 1536
        host_dim = self.host_model.config.hidden_size   # 2304
        
        print(f"Initializing Translator: {donor_dim} -> {host_dim}")

        cfg_dict = toml.load("../vec2vec/finetuning_unsupervised/online_flickr/config.toml")

        self.cfg = SimpleNamespace(**cfg_dict)

        model_path = os.path.join("../vec2vec/finetuning_unsupervised/online_flickr", 'model.pt')

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

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt"),
        max_tokens: int = Input(description="Max tokens", default=512),
    ) -> ConcatenateIterator[str]:

        torch.manual_seed(42)

        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].cuda()

        with torch.inference_mode():
            vision_tower = self.donor_model.get_model().get_vision_tower()
            image_forward = vision_tower.vision_tower(image_tensor, output_hidden_states=True)
            
            features = image_forward.hidden_states[-2][:, 1:] 
            
            visual_embeds_qwen = self.donor_model.get_model().mm_projector(features)

            print("Saving histogram of visual embeddings means across each token.")
            
            mean_embeds = visual_embeds_qwen.mean(dim=-1, keepdim=False).squeeze(0).detach().cpu().numpy()
            print(mean_embeds.shape)
            plt.hist(mean_embeds, bins=50, density=True, alpha=0.4)

            kde = gaussian_kde(mean_embeds)
            xs = np.linspace(mean_embeds.min(), mean_embeds.max(), 500)

            plt.title("Mean embedding distribution (LLaVA Qwen)")
            plt.plot(xs, kde(xs), linewidth=2)
            plt.savefig("histogram.png")
            
            print(visual_embeds_qwen.mean(), visual_embeds_qwen.min(), visual_embeds_qwen.max())
            
            print(f"Qwen Visual Shape: {visual_embeds_qwen.shape}")

            print("--- TRANSLATING ---")
            # 1536 -> 2304
           
            model_input = visual_embeds_qwen.to(torch.float32)

            ins = {'unsup_model': model_input}
            out_set = {'sup_model'}
            _, visual_embeds_gemma = self.translator(ins=ins, out_set=out_set)
            visual_embeds_gemma = visual_embeds_gemma["sup_model"]["unsup_model"]


            print(f"Translated embeddings shape: {visual_embeds_gemma.shape}")
            # --- NEW: Compute closest discretized tokens ---
            print("--- DISCRETIZING VISUAL EMBEDDINGS ---")
            # Get Gemma's embedding weights [vocab_size, hidden_size]
            gemma_weights = self.host_model.get_input_embeddings().weight.data

            # Compute cosine similarity between visual embeds and vocabulary
            # visual_embeds_gemma: [1, 256, 2304] -> [256, 2304]
            # Normalize weights and embeddings for cosine similarity
            v_norm = torch.nn.functional.normalize(visual_embeds_gemma.squeeze(0), p=2, dim=-1)
            w_norm = torch.nn.functional.normalize(gemma_weights, p=2, dim=-1)

            # Resulting similarity matrix: [256, vocab_size]
            similarities = torch.matmul(v_norm, w_norm.t())

            #dist = torch.cdist(visual_embeds_gemma.squeeze(0), gemma_weights, p=2)
            #closest_token_ids = torch.argmin(dist, dim=-1)

            # Get the indices of the highest similarity
            closest_token_ids = torch.argmax(similarities, dim=-1)

            # Decode these tokens to see what Gemma "sees"
            discretized_tokens = self.host_tokenizer.convert_ids_to_tokens(closest_token_ids)
            discretized_text = self.host_tokenizer.decode(closest_token_ids)

            print(f"Closest tokens (sample): {discretized_tokens[:10]}")
            print(f"Discretized representation: {discretized_text[:100]}...")

            print(visual_embeds_gemma.mean(), visual_embeds_gemma.min(), visual_embeds_gemma.max())
            
            print("--- STITCHING EMBEDDINGS ---")
            prompt_template = f"<bos><start_of_turn>user\nBelow is a visual signal (morse code) you have to decipher: {DEFAULT_IMAGE_TOKEN}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            #prompt_template = f"<bos><start_of_turn>user\n{DEFAULT_IMAGE_TOKEN}\n<end_of_turn>\n<start_of_turn>model\n"

            # Split the prompt into parts before and after the image token
            parts = prompt_template.split(DEFAULT_IMAGE_TOKEN)
            prefix_ids = self.host_tokenizer(parts[0], return_tensors='pt', add_special_tokens=False).input_ids.to(device)
            suffix_ids = self.host_tokenizer(parts[1], return_tensors='pt', add_special_tokens=False).input_ids.to(device)

            # Convert to embeddings
            prefix_embeds = self.host_model.get_input_embeddings()(prefix_ids)
            suffix_embeds = self.host_model.get_input_embeddings()(suffix_ids)

            # Scale visual_embeds_gemma to match prefix_embeds magnitude
            # This prevents Gemma from treating visual tokens as outliers/noise
            #scale_factor = prefix_embeds.abs().mean() / visual_embeds_gemma.abs().mean()
            #visual_embeds_gemma = visual_embeds_gemma * scale_factor

            # Concatenate: [Prefix] + [Visual] + [Suffix]
            inputs_embeds_manual = torch.cat([prefix_embeds, visual_embeds_gemma, suffix_embeds], dim=1)
            
            attention_mask_manual = torch.ones(inputs_embeds_manual.shape[:2], dtype=torch.long, device=device_map)
            
            dummy_input_ids = torch.zeros(inputs_embeds_manual.shape[:2], dtype=torch.long, device=device_map)
            seq_len = inputs_embeds_manual.shape[1]
            position_ids_manual = torch.arange(seq_len, dtype=torch.long, device=device_map)

            print(f"Final Input Shape: {inputs_embeds_manual.shape}")

            self.host_model.generation_config.max_length = inputs_embeds_manual.shape[1] + max_tokens

            if self.host_model.generation_config.pad_token_id is None:
                self.host_model.generation_config.pad_token_id = self.host_tokenizer.eos_token_id

            print("--- GENERATING (NATIVE GEMMA) ---")
            
            output_ids = self.host_model.generate(
                input_ids=dummy_input_ids,
                inputs_embeds=inputs_embeds_manual,
                attention_mask=attention_mask_manual,
                max_new_tokens=max_tokens,
                pad_token_id=self.host_tokenizer.pad_token_id,
                eos_token_id=self.host_tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                temperature=0.1, # 0.9
                repetition_penalty=1.2, # 1.2
                top_p=1.0 # 1.0
            )

            generated_ids = output_ids[0][dummy_input_ids.shape[1]:]

            # Decode
            text_output = self.host_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            print(f"OUTPUT: {text_output}")
            yield text_output
