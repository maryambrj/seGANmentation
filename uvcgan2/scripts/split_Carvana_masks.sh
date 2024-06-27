#!/bin/bash

# Directories containing the original masks and the split images
mask_dir="data/Carvana/all_annotated/masks"
train_img_dir="data/Carvana/train/images"
val_img_dir="data/Carvana/val/images"

# Directories to store the split masks
train_mask_dir="data/Carvana/train/masks"
val_mask_dir="data/Carvana/val/masks"

# Function to copy masks corresponding to a set of images
copy_masks() {
    local img_dir=$1
    local mask_target_dir=$2
    for img_file in $(ls $img_dir); do
        base_name=$(basename $img_file .jpg) # Extracts the name without the .jpg extension
        mask_file="${base_name}_mask.gif" # Constructs the mask filename
        cp "${mask_dir}/${mask_file}" "${mask_target_dir}/"
    done
}

# Copy masks corresponding to the train and validation image sets
copy_masks "$train_img_dir" "$train_mask_dir"
copy_masks "$val_img_dir" "$val_mask_dir"