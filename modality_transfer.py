import torch
import gc
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, LlavaProcessor, LlavaForConditionalGeneration
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Extract LLaVA Visual Embeddings
# ==========================================
print("--- STEP 1: Extracting Visual Embeddings ---")
llava_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(llava_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)

# Load your image
image_path = "image.jpg" # Ensure this path is correct
image = Image.open(image_path).convert("RGB")
inputs = processor(text="<image>", images=image, return_tensors="pt").to(device)

with torch.no_grad():
    vision_outputs = llava_model.model.vision_tower(inputs["pixel_values"], output_hidden_states=True)
    vision_hidden = vision_outputs.hidden_states[-1]
    visual_embeds_raw = llava_model.model.multi_modal_projector(vision_hidden)
    # Remove CLS token, keep spatial patches (usually 576 tokens)
    visual_embeds_raw = visual_embeds_raw[:, 1:, :]

visual_embeds_np = visual_embeds_raw[0].float().cpu().numpy()

# Free memory!
del llava_model
del processor
torch.cuda.empty_cache()
gc.collect()

# ==========================================
# 2. Train the Semantic Bridge
# ==========================================
print("\n--- STEP 2: Training the Semantic Bridge ---")
source_id = "lmsys/vicuna-7b-v1.5"
target_id = "Qwen/Qwen2.5-7B"

src_tok = AutoTokenizer.from_pretrained(source_id, use_fast=False)
tgt_tok = AutoTokenizer.from_pretrained(target_id)

print("Loading embeddings for alignment...")
src_embeds = AutoModelForCausalLM.from_pretrained(source_id, torch_dtype=torch.float16, device_map="cpu").get_input_embeddings().weight.detach().numpy()
tgt_embeds = AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=torch.float16, device_map="cpu").get_input_embeddings().weight.detach().numpy()

src_vocab = src_tok.get_vocab()
tgt_vocab = tgt_tok.get_vocab()
common_tokens = set(src_vocab.keys()).intersection(set(tgt_vocab.keys()))
anchors = [t for t in common_tokens if len(t) > 3][:10000]

X = np.array([src_embeds[src_vocab[t]] for t in anchors])
Y = np.array([tgt_embeds[tgt_vocab[t]] for t in anchors])

print("Fitting Ridge Regression Matrix...")
bridge = Ridge(alpha=0.1)
bridge.fit(X, Y)

# ==========================================
# 3. Transform Visual Tokens (Global Alignment)
# ==========================================
print("\n--- STEP 3: Translating Visual Tokens ---")
transformed_visual_np = bridge.predict(visual_embeds_np)

# ==========================================
# 4. Optimal Transport Barycentric Projection (Pure PyTorch)
# ==========================================
print("\n--- STEP 4: Optimal Transport (PyTorch Sinkhorn Log-Domain) ---")
vocab_subset_size = 20000

# Convert directly to PyTorch tensors for stable math (using float32 for stability vs memory tradeoff)
v_embeds_tensor = torch.tensor(transformed_visual_np, dtype=torch.float32)
t_embeds_tensor = torch.tensor(tgt_embeds[:vocab_subset_size], dtype=torch.float32)

# Normalize for Cosine distance
v_norm = torch.nn.functional.normalize(v_embeds_tensor, p=2, dim=1)
t_norm = torch.nn.functional.normalize(t_embeds_tensor, p=2, dim=1)

print("Computing Cost Matrix...")
# Compute Cosine Distance Cost Matrix and clamp for stability
C = 1.0 - torch.mm(v_norm, t_norm.T)
C = torch.clamp(C, min=0.0, max=2.0)

# Uniform marginal distributions in log space
N, M = C.shape
loga = torch.log(torch.ones(N, dtype=torch.float32) / N)
logb = torch.log(torch.ones(M, dtype=torch.float32) / M)

print("Solving Sinkhorn Optimal Transport (Log-Domain in PyTorch)...")
reg = 0.05
num_iters = 200

# Initialize dual variables in log space
u = torch.zeros_like(loga)
v = torch.zeros_like(logb)

# Pure PyTorch Log-Domain Sinkhorn Loop 
for _ in range(num_iters):
    # Update v
    v = logb - torch.logsumexp(-C / reg + u.unsqueeze(1), dim=0)
    # Update u
    u = loga - torch.logsumexp(-C / reg + v.unsqueeze(0), dim=1)

# Reconstruct the transport plan P from log space
log_P = -C / reg + u.unsqueeze(1) + v.unsqueeze(0)
P = torch.exp(log_P)

# Normalize the transport plan so rows sum to 1 (Conditional Probabilities)
P_cond = P / P.sum(dim=1, keepdim=True)

# THE MAGIC: Multiply the probability matrix by the actual target embeddings.
ot_continuous_visual_tensor = torch.mm(P_cond, t_embeds_tensor)
print(f"OT Transformed shape: {ot_continuous_visual_tensor.shape}")

# Cast back to numpy float16 to match the rest of your pipeline
ot_continuous_visual_np = ot_continuous_visual_tensor.numpy().astype(np.float16)

# Free memory
del src_embeds
del tgt_embeds
del t_embeds_tensor
del C, P, log_P, u, v
gc.collect()

# ==========================================
# 5. Inject into Base LLM & Generate
# ==========================================
print("\n--- STEP 5: Injecting & Generating Text ---")
target_model = AutoModelForCausalLM.from_pretrained(
    target_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)

# Convert OT numpy array back to tensor, matching target model's dtype
visual_tensors = torch.tensor(
    ot_continuous_visual_np,
    dtype=target_model.dtype,
    device=device
).unsqueeze(0)

text_prompt = "\nDetailed description of the image:"
text_inputs = tgt_tok(text_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    text_embeds = target_model.get_input_embeddings()(text_inputs.input_ids)

# --- THE CONTINUOUS INJECTION ---
inputs_embeds = torch.cat([visual_tensors, text_embeds], dim=1)

attention_mask = torch.ones(
    (1, inputs_embeds.shape[1]),
    dtype=torch.long,
    device=device
)

print("Generating response from Base LLM...")
with torch.no_grad():
    outputs = target_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tgt_tok.eos_token_id
    )

generated_text = tgt_tok.decode(outputs[0], skip_special_tokens=True)

print("\n==========================================")
print("🎯 BASE LLM OUTPUT (OT Injected):")
print("==========================================")
print(generated_text)
