import torch
import os
import requests
import copy
from PIL import Image
from io import BytesIO
from cog import BasePredictor, Input, Path, ConcatenateIterator

# --- LLaVA Imports ---
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

# --- Transformers Imports ---
from transformers import AutoConfig, LlamaForCausalLM, GenerationConfig, Qwen2ForCausalLM
from transformers.generation.utils import GenerationMixin
from transformers import TextIteratorStreamer

os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"
device_map = "cuda"

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
        
        print(f"Loading LLaVA Donor from {llava_path}...")
        self.tokenizer, self.llava_model, self.image_processor, _ = load_pretrained_model(
            llava_path, None, "llava-qwen", device_map=device_map
        )

        # --- 1. CLONE THE BRAIN (The Ultimate Fix) ---
        print("Cloning LLaVA Internal LLM...")
        
        # Get the config from the LIVE model
        # We assume LLaVA wrapped a Qwen model into a Llama Structure. 
        # We must use LlamaConfig to match the class structure.
        donor_config = self.llava_model.config
        print("----CONFIG----")
        print(donor_config)
        
        # Create a clean config for our manual model
        # We intentionally identify as "Llama" to match LLaVA's internal logic
        manual_config = copy.deepcopy(donor_config)
        manual_config.architectures = ["Qwen2ForCausalLM"] 
        manual_config.model_type = "llama"
        
        # Disable sliding window explicitly to avoid Llama/Qwen conflict
        #manual_config.sliding_window = None
        #manual_config.use_sliding_window = False
        
        # Force Eager attention to avoid kernel optimization bugs with hacked configs
        #manual_config._attn_implementation = "eager"

        # Instantiate an empty Llama Shell
        self.manual_model = Qwen2ForCausalLM(manual_config).to(dtype=torch.float16).cuda()
        
        # --- 2. DIRECT STATE DICT TRANSFER ---
        # Instead of loading from disk (which causes size mismatches and missing biases),
        # we copy the weights directly from the LLaVA object in memory.
        print("Copying Weights directly from memory...")
        
        # LLaVA structure: self.llava_model (LlavaLlamaForCausalLM) -> .model (LlavaLlamaModel) -> layers
        # Manual structure: self.manual_model (LlamaForCausalLM) -> .model (LlamaModel) -> layers
        # The keys usually align perfectly if we grab the state_dict of the inner model.
        
        # 1. Copy Backbone
        # We filter out vision_tower and mm_projector from the LLaVA state dict
        llava_state = self.llava_model.state_dict()
        manual_state = self.manual_model.state_dict()
        
        filtered_state = {}
        for k, v in llava_state.items():
            # Exclude vision/projector
            if "vision_tower" in k or "mm_projector" in k or "image_newline" in k:
                continue
            
            # Check if key exists in manual model
            if k in manual_state:
                filtered_state[k] = v
            else:
                # Handle potential naming mismatch if LLaVA wraps differently
                # But usually LlavaLlamaForCausalLM mimics LlamaForCausalLM exactly
                pass

        # Load weights
        missing, unexpected = self.manual_model.load_state_dict(filtered_state, strict=False)
        print(f"Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # --- 3. EXTRAS ---
        self.image_newline = getattr(self.llava_model.model, 'image_newline', None)
        
        # Match Generation Config
        self.manual_model.generation_config = GenerationConfig.from_model_config(manual_config)
        self.manual_model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        print("Setup Complete.")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="Top P", default=0.9),
        temperature: float = Input(description="Temperature", default=0.7),
        max_tokens: int = Input(description="Max tokens", default=512),
    ) -> ConcatenateIterator[str]:
        
        torch.manual_seed(42)
        
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
        
        # =========================================================================
        # A. GOLD STANDARD
        # =========================================================================
        print("\n--- GOLD STANDARD ---")
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        input_ids = tokenizer_image_token(inp, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        with torch.inference_mode():
             # We capture the position_ids LLaVA generates!
             _, position_ids_gold, _, _, inputs_embeds_gold, _ = self.llava_model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensor
            )
             print(f"Position Ids: {position_ids_gold}")
        
        print(f"Gold Shape: {inputs_embeds_gold.shape}")
        
        gen_kwargs = dict(
            do_sample=False, 
            max_new_tokens=128, 
            use_cache=True,
            top_p=1.0, temperature=1.0, repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_ids_gold = GenerationMixin.generate(self.llava_model, inputs_embeds=inputs_embeds_gold, **gen_kwargs)
        text_gold = self.tokenizer.decode(output_ids_gold[0], skip_special_tokens=True).strip()
        print(f"GOLD: {text_gold}")

        # =========================================================================
        # B. MANUAL REPLICATION
        # =========================================================================
        print("\n--- MANUAL REPLICATION ---")
        with torch.inference_mode():
            # 1. Vision (Bypass Wrapper)
            wrapper = self.llava_model.get_model().get_vision_tower()
            hf_vision_model = wrapper.vision_tower 
            image_forward_out = hf_vision_model(image_tensor, output_hidden_states=True)
            
            selected_layer = image_forward_out.hidden_states[-2]
            features_no_cls = selected_layer[:, 1:] 
            visual_embeddings = self.llava_model.get_model().mm_projector(features_no_cls)

            # 2. Newline
            if self.image_newline is not None:
                newline_embed = self.image_newline[None, None, :].to(dtype=visual_embeddings.dtype, device=visual_embeddings.device)
                visual_embeddings = torch.cat([visual_embeddings, newline_embed], dim=1)
            
            # 3. Text
            text_str = ""#"\n" + prompt
            text_inputs = self.tokenizer(text_str, return_tensors='pt', add_special_tokens=False).to(device_map)
            text_embeddings = self.manual_model.get_input_embeddings()(text_inputs.input_ids)
            
            # 4. Concat
            visual_embeddings = visual_embeddings.to(dtype=text_embeddings.dtype)
            inputs_embeds_manual = torch.cat([visual_embeddings, text_embeddings], dim=1)
            
            # MSE Check
            mse = torch.nn.functional.mse_loss(inputs_embeds_gold.float(), inputs_embeds_manual.float())
            print(f"MSE: {mse.item():.9f}")

            # 5. Generate
            print("Generating Manual Text...")
            attention_mask_manual = torch.ones(inputs_embeds_manual.shape[:2], dtype=torch.long, device=device_map)
            
            # CRITICAL: We pass the EXACT position_ids from the Gold Run
            # This ensures the model knows the image isn't just text tokens 0-256

            
            output_ids_manual = GenerationMixin.generate(
                self.manual_model, 
                inputs_embeds=inputs_embeds_manual, 
                attention_mask=attention_mask_manual,
                position_ids=position_ids_gold, # <--- THE MISSING KEY
                **gen_kwargs
            )
            text_manual = self.tokenizer.decode(output_ids_manual[0], skip_special_tokens=True).strip()
            print(f"MANUAL: {text_manual}")
            
            if text_gold == text_manual:
                print("\n>>> VICTORY: The outputs are IDENTICAL.")
                yield f"SUCCESS: {text_manual}"
            else:
                print(f"\n>>> MISMATCH: '{text_gold}' vs '{text_manual}'")
                yield f"MISMATCH: {text_manual}"
