#!/bin/bash

# Number of images to randomly select
N=100

# Directory containing the images and masks
IMAGES_DIR="./data/Carvana_resized/val/images"
MASKS_DIR="./data/Carvana_resized/val/masks"

# Destination directories
DEST_IMAGES_DIR="./data/Carvana_resized_10percent/val/images"
DEST_MASKS_DIR="./data/Carvana_resized_10percent/val/masks"

# Randomly select N files from the images directory, maintaining the order
ls "$IMAGES_DIR" | sort -R | head -n $N | while read filename; do
    # Copy the images to the new location
    cp "$IMAGES_DIR/$filename" "$DEST_IMAGES_DIR"
    
    # Generate the mask filename based on the image filename
    mask_filename="${filename%.jpg.png}_mask.gif.png"

    # Copy the corresponding masks to the new location
    cp "$MASKS_DIR/$mask_filename" "$DEST_MASKS_DIR"
done