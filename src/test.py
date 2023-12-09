import torch
import pandas as pd
import transformers

def main():
    print("PyTorch version:", torch.__version__)
    print("Pandas version:", pd.__version__)
    print("HuggingFace Version:" , transformers.__version__)


if __name__ == "__main__":
    main()

