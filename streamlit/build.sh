#!/bin/bash

# Variables
IMAGE_NAME="kvasir_demo"
TAG="latest"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image '${IMAGE_NAME}:${TAG}' built successfully."
else
    echo "Docker image build failed."
    exit 1
fi
