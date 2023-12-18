# coding: utf-8
"""
This module contains all configurations for the project.
"""


from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

MODEL_CHOICE = "openai/clip-vit-base-patch32"
MODEL = CLIPModel.from_pretrained(MODEL_CHOICE)
PROCESSOR = CLIPProcessor.from_pretrained(MODEL_CHOICE)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHOICE)


FOLDER = "samples"
