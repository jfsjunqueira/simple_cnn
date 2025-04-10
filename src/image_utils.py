import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import os
import json
from typing import Tuple, Dict, Optional, List
from .config import config

class ImageDataset(Dataset):
    """Custom Dataset for loading images with their labels."""
    
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df
        self.transform = transform or transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create label mapping
        unique_labels = sorted(df['class_name'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['class_name']
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return image_tensor, label_idx

def save_split_indices(train_idx: List[int], val_idx: List[int], test_idx: List[int], 
                      save_dir: str):
    """Save dataset split indices to a JSON file"""
    split_info = {
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx
    }
    
    os.makedirs(save_dir, exist_ok=True)
    split_file = os.path.join(save_dir, 'dataset_splits.json')
    
    with open(split_file, 'w') as f:
        json.dump(split_info, f)
    
    return split_file

def load_split_indices(split_file: str) -> Tuple[List[int], List[int], List[int]]:
    """Load dataset split indices from a JSON file"""
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    
    return (split_info['train_indices'], 
            split_info['val_indices'], 
            split_info['test_indices'])

def create_data_loaders(split_file: Optional[str] = None, save_dir: Optional[str] = None
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from the CSV file.
    
    Args:
        split_file: Optional path to a JSON file containing split indices
        save_dir: Optional directory to save split indices if not loading from file
    
    Returns:
        Tuple containing train, validation, and test DataLoader objects
    """
    # Read the CSV file
    df = pd.read_csv(config.data_csv_path)
    
    # Create dataset
    full_dataset = ImageDataset(df)
    
    if split_file and os.path.exists(split_file):
        # Load existing splits
        train_idx, val_idx, test_idx = load_split_indices(split_file)
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)
    else:
        # Create new splits
        total_size = len(full_dataset)
        train_size = int(total_size * config.train_split)
        val_size = int(total_size * config.val_split)
        test_size = total_size - train_size - val_size
        
        # Split the dataset with fixed seed
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Save split indices if save_dir is provided
        if save_dir:
            split_file = save_split_indices(
                train_dataset.indices,
                val_dataset.indices,
                test_dataset.indices,
                save_dir
            )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def load_and_transform_image(image_path: str) -> torch.Tensor:
    """
    Load an image from path and transform it into a tensor suitable for the CNN.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        torch.Tensor: Transformed image tensor of shape (1, channels, height, width)
    """
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((config.image_height, config.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)  # Shape: (1, channels, height, width)

def get_class_names() -> Dict[int, str]:
    """
    Get a mapping of class indices to class names.
    
    Returns:
        Dict mapping class indices to their names
    """
    df = pd.read_csv(config.data_csv_path)
    unique_labels = sorted(df['class_name'].unique())
    return {idx: label for idx, label in enumerate(unique_labels)}