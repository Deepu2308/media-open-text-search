import torch
import pandas as pd
from transformers import pipeline

def main():
    print("PyTorch version:", torch.__version__)
    print("Pandas version:", pd.__version__)

    unmasker = pipeline('fill-mask', model='bert-base-uncased')
    output = unmasker("The installation worked and i feel [MASK] about it!!")[0]['sequence'].upper()
    print(output)

if __name__ == "__main__":
    main()

