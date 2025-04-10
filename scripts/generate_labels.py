import pandas as pd
import os
from pathlib import Path
import argparse

def generate_labels(img_folder: str, output_path: str):
    """
    Generate a CSV file with labels for all images in the given folder.
    
    Args:
        img_folder (str): Path to the folder containing the images.
        output_path (str): Path to the output CSV file.
    """
    base_dir = Path(img_folder)
    # List of all sub-directories (classes)
    sub_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    # Now for each of the sub-directories, we will create a list of all the images in that directory with their paths and classes (name of the directories they are in)
    image_paths = []
    image_classes = []
    for sub_dir in sub_dirs:
        # Get the class name from the directory name
        class_name = sub_dir.name
        
        # List all image files in the directory
        for img_file in sub_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(img_file.absolute())
                image_classes.append(class_name)
    # Create a DataFrame with the image paths and classes
    df = pd.DataFrame({
        'image_path': image_paths,
        'class_name': image_classes
    })
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Labels saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate labels for images')
    parser.add_argument('--img_folder', type=str, required=True, help='Path to the folder containing the images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output CSV file')
    args = parser.parse_args()
    generate_labels(img_folder=args.img_folder, output_path=args.output_path)
