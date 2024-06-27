#!/bin/bash

# Get the base directory relative to the script location
BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

# Temporary directory for storing base names files
TMP_DIR="$BASE_DIR/tmp"
mkdir -p "$TMP_DIR"  # Ensure temporary directory exists

DIR1="$BASE_DIR/data/Carvana_resized_10percent/train/masks"
DIR2="$BASE_DIR/data/Carvana_resized_unsupervised_10percent/train/masks"

# Generate base names list for DIR1
cd "$DIR1"
ls | grep -oP '.*(?=_[0-9]{2}_mask\.gif\.png)' | sort -u > "$TMP_DIR/dir1_basenames.txt"

# Generate base names list for DIR2
cd "$DIR2"
ls | grep -oP '.*(?=_[0-9]{2}_mask\.gif\.png)' | sort -u > "$TMP_DIR/dir2_basenames.txt"

# Compare the lists and output common names
comm -12 "$TMP_DIR/dir1_basenames.txt" "$TMP_DIR/dir2_basenames.txt" > "$TMP_DIR/common_basenames.txt"

echo "Common base names (potential conflicts):"
cat "$TMP_DIR/common_basenames.txt"
echo "Number of common base names (potential conflicts):"
wc -l "$TMP_DIR/common_basenames.txt"
