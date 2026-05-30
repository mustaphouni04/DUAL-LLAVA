import torch, math
from translatorv3 import Predictor, load_image
from pathlib import Path

predictor = Predictor()
predictor.setup()  

result_iterator = predictor.predict(
    image="../cc3m_filtered/flickr30k-images/427936315.jpg",
    prompt="",
    #top_p=1.0,
    #temperature=0.1,
    max_tokens=300,
    )

output_text = "".join(list(result_iterator))
print(output_text)
