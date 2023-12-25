# Variables
IMAGE_NAME := media-ots-image
CONTAINER_NAME := media-ots-container
HOST_DATA_DIR := $(shell pwd)/src
CONTAINER_DATA_DIR := /src

.PHONY: all build run clean

all: build

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container with Jupyter Notebook and persist the data
bash:
	docker run --rm -p 8888:8888 -v "$(HOST_DATA_DIR):$(CONTAINER_DATA_DIR)" -it --name $(CONTAINER_NAME) media-ots-image  bash

# Start the Jupyter Notebook server
jupyter:
	docker run --rm -p 8888:8888 -v "$(HOST_DATA_DIR):$(CONTAINER_DATA_DIR)" -it --name $(CONTAINER_NAME) media-ots-image jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Start the Gradio interface
interface:
	docker run --rm -p 8888:8888 -v "$(HOST_DATA_DIR):$(CONTAINER_DATA_DIR)" -it --name $(CONTAINER_NAME) $(IMAGE_NAME) python $(CONTAINER_DATA_DIR)/interface.py

# Remove the Docker container (if it exists) and image
clean:
	-docker rm -f $(CONTAINER_NAME)
	docker rmi $(IMAGE_NAME)