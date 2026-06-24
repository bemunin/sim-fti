#!/bin/bash

# Ensure that if a command in a pipeline fails, the whole pipeline fails
set -o pipefail

# Define the target directory
TARGET_DIR="/media/bemunin/nv-fti-course/ros-docker-images"

# List of Docker images to save
IMAGES=(
    "nv_fti_ros_humble:latest"
    "isaac_sim_ros:ubuntu_22_humble"
)

# 1. Create the target directory if it doesn't already exist
echo "Checking target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# 2. Loop through the array and save each image
echo "Starting Docker image backup (with gzip compression)..."
echo "-----------------------------------"

for IMAGE in "${IMAGES[@]}"; do
    # Replace colons and slashes with underscores to create a safe filename
    SAFE_FILENAME=$(echo "$IMAGE" | tr ':/' '__')
    
    # Update extension to .tar.gz
    OUTPUT_FILE="${TARGET_DIR}/${SAFE_FILENAME}.tar.gz"

    echo "Saving and compressing $IMAGE..."
    
    # Execute the docker save command and pipe it directly into gzip
    docker save "$IMAGE" | gzip > "$OUTPUT_FILE"

    # Check if the pipeline was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully saved to: $OUTPUT_FILE"
    else
        echo "❌ Failed to save $IMAGE. Please check if the image exists locally."
        # Remove the corrupted/empty file if it failed
        rm -f "$OUTPUT_FILE"
    fi
    echo "-----------------------------------"
done

echo "Backup complete! You can find your compressed images in $TARGET_DIR"