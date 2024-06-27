#!/bin/bash

# Get the base directory relative to the script location
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

# Directories containing the masks
DIR1="$BASE_DIR/data/Carvana_resized_10percent/train/masks"
DIR2="$BASE_DIR/data/Carvana_resized/train/masks"
NEW_MASKS_SUBSET_DIR="$BASE_DIR/data/Carvana_resized_unsupervised_10percent/train/masks"

# Temporary directory for storing base names files and processing
TMP_DIR="$BASE_DIR/tmp"
mkdir -p "$TMP_DIR"  # Ensure temporary directory exists

# Extract base names and remove duplicates from DIR1
cd "$DIR1"
ls | grep -oP '.*(?=_[0-9]{2}_mask\.gif\.png)' | sort -u > "$TMP_DIR/exclude_basenames.txt"

# Generate exclusion patterns based on DIR1 base names
cd "$DIR2"
ls *.gif.png | grep -oP '.*(?=_[0-9]{2}_mask\.gif\.png)' | sort -u > "$TMP_DIR/dir2_all_basenames.txt"
grep -vFf "$TMP_DIR/exclude_basenames.txt" "$TMP_DIR/dir2_all_basenames.txt" > "$TMP_DIR/eligible_basenames.txt"

# Debug output
echo "Eligible base names:"
cat "$TMP_DIR/eligible_basenames.txt"

# Select random samples from eligible basenames and ensure no overlap with DIR1
shuf -n 400 "$TMP_DIR/eligible_basenames.txt" | while read base; do
    echo "Checking files for base name: $base"
    # Check each file existence before attempting to list or copy
    for i in {01..16}; do
        if [ -f "${base}_${i}_mask.gif.png" ]; then
            echo "${base}_${i}_mask.gif.png exists"
            echo "${base}_${i}_mask.gif.png" >> "$TMP_DIR/final_selection.txt"
        else
            echo "${base}_${i}_mask.gif.png does not exist"
        fi
    done
done

echo "Selected files:"
cat "$TMP_DIR/final_selection.txt"

# Ensure the destination directory exists
mkdir -p "$NEW_MASKS_SUBSET_DIR"

# Copy the selected masks to the new folder
cat "$TMP_DIR/final_selection.txt" | xargs -I {} cp {} "$NEW_MASKS_SUBSET_DIR/"

echo "Files copied to new directory. No overlapping base names with DIR1."
