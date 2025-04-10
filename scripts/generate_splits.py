import torch
from torch.utils.data import random_split
import pandas as pd
import json
import os
import sys
from src.config import config
from src.image_utils import ImageDataset

def generate_splits_file(model_dir: str):
    """Generate dataset splits file using the same seed and ratios used in training"""
    
    # Read the CSV file and create dataset
    print("\nReading dataset...")
    df = pd.read_csv(config.data_csv_path)
    full_dataset = ImageDataset(df)
    
    # Calculate split sizes (same as in original training)
    total_size = len(full_dataset)
    train_size = int(total_size * config.train_split)
    val_size = int(total_size * config.val_split)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset size: {total_size}")
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")
    print(f"Test size: {test_size}")
    
    # Create splits using the same seed
    generator = torch.Generator().manual_seed(42)  # Same seed used in training
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create splits info
    split_info = {
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'test_indices': test_dataset.indices
    }
    
    # Save to file
    splits_file = os.path.join(model_dir, 'dataset_splits.json')
    with open(splits_file, 'w') as f:
        json.dump(split_info, f)
    
    print(f"\nSplit indices saved to: {splits_file}")
    return splits_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_splits.py <model_directory>")
        sys.exit(1)
        
    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)
        
    splits_file = generate_splits_file(model_dir)
    print("\nNow you can run the evaluation with:")
    print(f"python -m scripts.evaluate {os.path.join(model_dir, 'model.pth')} --split-file {splits_file}")