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


def image_finder(user_query: str, n: int = 1):
    """
    This function takes a query and returns the best matching image.
    """

    print("Finding meme for query: ", user_query)

    user_query = featurize_query(user_query)

    scores, matches = image_dataset["train"].get_nearest_examples(
        "features", user_query, k=n
    )

    matches = matches["image"]
    if n == 1:
        return matches[0].resize((300, 300))

    else:
        return scores, matches
