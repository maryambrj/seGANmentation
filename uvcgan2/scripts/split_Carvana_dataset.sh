#!/bin/bash

# Directory containing the images
img_dir="data/Carvana/all_annotated/images"

# Directories for train and validation sets
train_dir="data/Carvana/train/images"
val_dir="data/Carvana/val/images"

# Extract unique car IDs and shuffle
ids=$(ls $img_dir | cut -d '_' -f 1 | sort | uniq | shuf)

# Calculate the number of cars for the validation set (20%)
total=$(echo "$ids" | wc -l)
val_size=$(($total * 20 / 100))

# Split IDs into train and validation sets
val_ids=$(echo "$ids" | head -n $val_size)
train_ids=$(echo "$ids" | tail -n +$(($val_size + 1)))

# Function to move files
move_files() {
    local id_list=$1
    local target_dir=$2
    for id in $id_list; do
        cp "${img_dir}/${id}_"* "${target_dir}/"
    done
}

# Move files to train and validation sets
move_files "$train_ids" "$train_dir"
move_files "$val_ids" "$val_dir"
