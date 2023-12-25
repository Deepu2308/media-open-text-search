# coding: utf-8
"""
This module starts an interface to search for images.
"""

import gradio as gr
from user_query import (
    image_finder,
)  # Assuming image_finder is the function you mentioned


def search_images(query: str):
    """
    This function takes a query and returns the best matching images.
    """
    # Call your image_finder function from user_query
    _, image_objects = image_finder(query, n=5)

    # Assuming image_finder now returns a list of PIL.Image objects
    return image_objects


# Define your Gradio interface
iface = gr.Interface(
    fn=search_images,  # the function to call
    inputs=gr.Textbox(
        lines=2, placeholder="Enter your search query here..."
    ),  # input type and properties
    outputs=[
        gr.Image(label="Result 1"),
        gr.Image(label="Result 2"),
        gr.Image(label="Result 3"),
        gr.Image(label="Result 4"),
        gr.Image(label="Result 5"),
    ],  # output type and properties
    title="Image Search",
    description="Enter a query to search for images.",
)

# Launch the application on port 8888
if __name__ == "__main__":
    iface.launch(server_port=8888, server_name="0.0.0.0")
