from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import warnings

warnings.filterwarnings("ignore")

pretrained = "scripts/checkpoints/llava-Qwen2.5-1.5B-Instruct-pretrain/checkpoint-34883"
model_name = "llava-qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map
)

model.eval()

url = "https://blog.sothebysrealty.co.uk/hs-fs/hubfs/Most%20Popular%20Dog%20Breeds%20UK-jpg.jpeg?width=1600&height=914&name=Most%20Popular%20Dog%20Breeds%20UK-jpg.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "plain"
conv = copy.deepcopy(conv_templates[conv_template])
conv.messages=[]

question = DEFAULT_IMAGE_TOKEN + "\nDescribe this image:"
print(DEFAULT_IMAGE_TOKEN in tokenizer.get_vocab())


conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    max_new_tokens=128,
    use_cache=False,
    eos_token_id=tokenizer.eos_token_id
)

text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
