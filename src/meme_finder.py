import io

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def preprocess_images(examples):
    images = examples["image"]
    processed_images = processor(images=images, return_tensors="pt")
    return processed_images


def extract_image_features(examples):
    # Extract the pixel_values tensor from the examples
    pixel_values = torch.Tensor(examples["pixel_values"]).float()

    # Get image features using the pixel_values tensor
    outputs = model.get_image_features(pixel_values)

    return {"features": outputs}


def featurize_images(folder="samples"):
    print("load image dataset")
    image_dataset = load_dataset("imagefolder", data_dir=f"data/{folder}")

    print("preprocess images")
    image_dataset = image_dataset.map(preprocess_images, batched=True, batch_size=4)

    print("extract image features")
    image_dataset = image_dataset.map(
        extract_image_features, batched=True, batch_size=4
    )

    image_features = np.array(image_dataset["train"]["features"])

    return image_features, image_dataset


def featurize_query(q):
    inputs = tokenizer([q], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)

    return text_features.detach().numpy()


image_features, image_dataset = featurize_images(folder="sample")


def meme_finder(q, folder="samples"):
    print("Finding meme for query: ", q)
    scores = np.matmul(featurize_query(q), image_features.T)
    best_match = np.argmax(scores)

    image = image_dataset["train"]["image"][best_match]

    # Example: Resize the image to 300x300 pixels
    new_size = (300, 300)
    resized_img = image.resize(new_size)

    return resized_img
