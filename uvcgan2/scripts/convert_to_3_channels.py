import os
import sys
from PIL import Image
import numpy as np

def convert_to_3_channel(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.gif')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')  # Ensure it's grayscale
            img_3_channel = np.stack((img,)*3, axis=-1)  # Duplicate channels
            img_3_channel = Image.fromarray(img_3_channel)
            img_3_channel.save(os.path.join(output_dir, filename))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_3_channel.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_to_3_channel(input_dir, output_dir)
