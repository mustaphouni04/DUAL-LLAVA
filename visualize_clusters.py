import torch
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import umap
from sklearn.preprocessing import normalize

# --------------------------------------------------
# 1. Setup and Model Loading
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/llava-1.5-7b-hf"

print(f"Loading model: {model_id}...")
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

model.eval()

# --------------------------------------------------
# 2. Load Image and Prepare Inputs
# --------------------------------------------------
# Note: Update this path to your specific image file
image_path = "../cc3m_filtered/flickr30k-images/3472419481.jpg"
image = Image.open(image_path).convert("RGB")

prompt = "Describe the image."
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# --------------------------------------------------
# 3. Extract Projected Visual Embeddings
# --------------------------------------------------
print("Extracting visual embeddings...")
with torch.no_grad():
    vision_outputs = model.model.vision_tower(
        inputs["pixel_values"],
        output_hidden_states=True
    )
    # Get last layer of vision tower
    vision_hidden = vision_outputs.hidden_states[-1]
    
    # Project vision features into the LLM's language space
    visual_embeds_raw = model.model.multi_modal_projector(vision_hidden)
    
    # Remove the CLS token (usually index 0) to keep only spatial tokens
    visual_embeds_raw = visual_embeds_raw[:, 1:, :]

# Convert to float32 for CPU-based UMAP/Math operations
visual_embeds_np = visual_embeds_raw[0].float().cpu().numpy()

# --------------------------------------------------
# 4. Nearest Neighbor Decoding (Visual -> Vocab)
# --------------------------------------------------
print("Decoding visual tokens to nearest vocabulary neighbors...")
with torch.no_grad():
    full_vocab_embeds = model.model.language_model.embed_tokens.weight.float().cpu().numpy()

# Normalize both sets for Cosine Similarity (Dot product of normalized vectors)
norm_visual = normalize(visual_embeds_np)
norm_vocab = normalize(full_vocab_embeds)

# Calculate similarity and find nearest neighbors
similarities = np.matmul(norm_visual, norm_vocab.T)
closest_token_ids = np.argmax(similarities, axis=1)

# Robust Decoding Loop to avoid NoneType/AttributeErrors
clean_tokens = []
for tid in closest_token_ids:
    # Use decode() for reliability; handle edge cases where tid might be None or out of range
    try:
        token_str = processor.tokenizer.decode([tid], skip_special_tokens=False).strip()
        if not token_str:
            # Fallback to the raw token representation if decode yields an empty string
            token_str = str(processor.tokenizer.convert_ids_to_tokens(int(tid)))
    except:
        token_str = "[UNK]"
    
    clean_tokens.append(token_str if token_str is not None else "[NULL]")

# Prepare the decoding string for the plot (Top 30 tokens)
decoded_str = " | ".join(clean_tokens[:30])
print(f"Sample of decoded visual tokens: {decoded_str}")

# --------------------------------------------------
# 5. UMAP Dimensionality Reduction
# --------------------------------------------------
print("Running UMAP (this may take a moment)...")
np.random.seed(42)
num_vocab_samples = 3000
indices = np.random.choice(full_vocab_embeds.shape[0], size=num_vocab_samples, replace=False)
vocab_sample = norm_vocab[indices]

# Stack visual and vocab tokens for a shared projection space
combined = np.vstack([norm_visual, vocab_sample])
labels = ["visual"] * len(norm_visual) + ["vocab"] * len(vocab_sample)

reducer = umap.UMAP(
    n_neighbors=25, 
    min_dist=0.1, 
    metric="cosine", 
    random_state=42
)
embedding_2d = reducer.fit_transform(combined)

# --------------------------------------------------
# 6. Final Plotting
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 11))

for t, color in zip(["visual", "vocab"], ["red", "blue"]):
    idx = [i for i, l in enumerate(labels) if l == t]
    ax.scatter(
        embedding_2d[idx, 0],
        embedding_2d[idx, 1],
        label=t,
        alpha=0.5,
        s=12,
        c=color
    )

ax.legend(loc='upper right')
ax.set_title("LLaVA 1.5 - Visual Embedding Space vs. Core Vocabulary", fontsize=14)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

# Annotate with decoded text at the bottom
wrapped_text = textwrap.fill(f"Nearest Vocab Neighbors (First 30 Visual Tokens): {decoded_str}", width=110)
plt.figtext(0.5, 0.03, wrapped_text, wrap=True, horizontalalignment='center', 
            fontsize=9, family='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver'))

plt.subplots_adjust(bottom=0.18) # Allocate space for the caption
plt.savefig("llava_modality_gap_decoded.png", dpi=300)
plt.show()

print("Process finished. Plot saved as 'llava_modality_gap_decoded.png'.")
