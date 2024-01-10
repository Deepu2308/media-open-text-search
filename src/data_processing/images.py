# coding: utf-8
"""
This module contains functions to load and process images.
"""

import faiss
import torch
from datasets import load_dataset, load_from_disk

from config import MODEL, PROCESSOR, N_CPU


def preprocess_images(examples):
    """
    This function takes a batch of image examples and returns the processed images.
    """
    images = examples["image"]
    processed_images = PROCESSOR(images=images, return_tensors="pt")
    return processed_images


def extract_image_features(examples):
    """
    This function takes a batch of image examples and returns the image features.
    """
    # Extract the pixel_values tensor from the examples
    pixel_values = torch.Tensor(examples["pixel_values"]).float()

    # Get image features using the pixel_values tensor
    outputs = MODEL.get_image_features(pixel_values)

    return {"features": outputs}


def get_image_dataset(folder="samples"):
    """
    This function loads the image dataset and returns it.
    """

    try:
        image_dataset = load_from_disk(f"data/processed/{folder}")
        print("loaded processed image dataset")
    except FileNotFoundError:
        print("load image dataset")
        image_dataset = load_dataset("imagefolder", data_dir=f"data/{folder}")

        print("preprocess images")
        image_dataset = image_dataset.map(
            preprocess_images, batched=True, batch_size=N_CPU
        )

        print("extract image features")
        image_dataset = image_dataset.map(
            extract_image_features, batched=True, batch_size=N_CPU
        )

        print("save dataset to disc")
        image_dataset.save_to_disk(f"data/processed/{folder}")
    except Exception as e:
        print(f"An error occurred: {e}")

    image_dataset["train"].add_faiss_index(
        column="features", metric_type=faiss.METRIC_INNER_PRODUCT
    )

    return image_dataset
