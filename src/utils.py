# coding=utf-8
"""
This module contains helper functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_image_matches(images, scores):
    """
    This function takes a list of PIL images and their corresponding scores and plots them.
    """
    # Create a figure and a set of subplots
    _, axs = plt.subplots(
        2, round(len(images) / 2 + 0.1), figsize=(15, 10)
    )  # Adjust the figsize as needed

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Iterate over images and their corresponding axes
    for ax, img, val in zip(axs, images, scores):
        # Convert the image to an array and display it
        ax.imshow(np.array(img))
        ax.axis("off")  # Hide axes
        ax.set_title(f"Score: {val:.2f}", fontsize=10)

    plt.tight_layout()
    plt.show()
