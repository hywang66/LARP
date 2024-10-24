#!/bin/bash

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

# Set the dataset paths, modify the source paths to your own dataset paths!

# UCF101
ln -s path/to/UCF101/videos ./data/ucf101

# Kinetics-600
ln -s path/to/kinetics600 ./data/k600

