from transformers import AutoConfig

LLAVA_PATH = "scripts/checkpoints/llava-Qwen2.5-1.5B-Instruct-pretrain/checkpoint-34883"
ORIGINAL_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"

conf_l = AutoConfig.from_pretrained(LLAVA_PATH)
conf_q = AutoConfig.from_pretrained(ORIGINAL_QWEN)

print(f"LLaVA Theta: {getattr(conf_l, 'rope_theta', 'N/A')}")
print(f"Qwen  Theta: {getattr(conf_q, 'rope_theta', 'N/A')}")

print(f"LLaVA Max Pos: {conf_l.max_position_embeddings}")
print(f"Qwen  Max Pos: {conf_q.max_position_embeddings}")
