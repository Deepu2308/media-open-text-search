# coding: utf-8
"""
This module contains finder function which takes a query and returns the best matching media.
"""


import numpy as np

from config import FOLDER, MODEL, TOKENIZER
from data_processing import get_image_dataset

image_dataset = get_image_dataset(folder=FOLDER)
image_features = np.array(image_dataset["train"]["features"])


def featurize_query(q: str):
    """
    This function takes a query and returns the features of the query.
    """

    inputs = TOKENIZER([q], padding=True, return_tensors="pt")
    text_features = MODEL.get_text_features(**inputs)

    return text_features.detach().numpy()


def image_finder(user_query: str):
    """
    This function takes a query and returns the best matching image.
    """

    print("Finding meme for query: ", user_query)
    scores = np.matmul(featurize_query(user_query), image_features.T)
    best_match = np.argmax(scores)

    image = image_dataset["train"]["image"][best_match]

    # Example: Resize the image to 300x300 pixels
    new_size = (300, 300)
    resized_img = image.resize(new_size)

    return resized_img
