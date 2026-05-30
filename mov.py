import torch, math
from predict import Predictor, load_image
from pathlib import Path

predictor = Predictor()
predictor.setup()  

result_iterator = predictor.predict(
    image="https://criver.widen.net/content/tphknpjpnk/jpeg/RM-001678.jpg",
    prompt="Describe this image in detail.",
    top_p=1.0,
    temperature=0.1,
    max_tokens=128,
    )

output_text = "".join(list(result_iterator))
print(output_text)
