from shared_predict import Predictor
import os 

predictor = Predictor()
predictor.setup()

response_generator = predictor.predict(
    image="https://facts.net/wp-content/uploads/2015/07/gorilla-3928903_1920.jpg",
    prompt="",
    model_choice="qwen",
    top_p=1.0,
    temperature=0.1,
    max_tokens=50,
)

full_response = ""
for text_chunk in response_generator:
    print(text_chunk, end="", flush=True)
    full_response += text_chunk

print() 
