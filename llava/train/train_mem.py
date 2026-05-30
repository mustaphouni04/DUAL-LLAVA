import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from llava.train.train import train, train_mm_projector

if __name__ == "__main__":
    train_mm_projector(attn_implementation="eager")
