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
run:
	docker run --rm -p 8888:8888 -v "$(HOST_DATA_DIR):$(CONTAINER_DATA_DIR)" -it --name $(CONTAINER_NAME) media-ots-image  bash

# Remove the Docker container (if it exists) and image
clean:
	-docker rm -f $(CONTAINER_NAME)
	docker rmi $(IMAGE_NAME)