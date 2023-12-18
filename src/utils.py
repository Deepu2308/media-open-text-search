import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Assuming you have a list of 6 PIL JpegImageFile objects named 'images'
# images = [image1, image2, image3, image4, image5, image6]


def plot_image_matches(images, scores):
    """
    This function takes a list of PIL images and their corresponding scores and plots them.
    """
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(
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
