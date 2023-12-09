# Variables
IMAGE_NAME := image_pytorch_hf
CONTAINER_NAME := container_pytorch_hf
HOST_DATA_DIR := $(shell pwd)/project
CONTAINER_DATA_DIR := /src/project

.PHONY: all build run clean

all: build

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container with Jupyter Notebook and persist the data
run:
	docker run -it --rm -p 8888:8888 -v "$(HOST_DATA_DIR):$(CONTAINER_DATA_DIR)" --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Remove the Docker container (if it exists) and image
clean:
	-docker rm -f $(CONTAINER_NAME)
	docker rmi $(IMAGE_NAME)