#!/bin/bash

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

# Set the dataset paths, modify the source paths to your own dataset paths!

# UCF-101
ln -s /mnt/bn/zilongdata-hl/dataset/UCF-101 ./data/ucf101

# # Kinetics-600
# ln -s path/to/kinetics600 ./data/k600

