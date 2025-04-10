import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import os
import json
from typing import Tuple, Dict, Optional, List, Callable
from .config import config
from .data_augmentation import DataAugmentation

def validate_data_csv():
    """
    Validate that the data CSV file exists and has the expected format.
    
    Returns:
        bool: True if valid, raises exception otherwise
    """
    # Check if file exists
    if not os.path.exists(config.data_csv_path):
        raise FileNotFoundError(f"Data CSV file not found at {config.data_csv_path}. "
                               f"Please ensure the data directory contains {config.data_csv_filename}")
    
    # Check if file is readable and has the expected format
    try:
        df = pd.read_csv(config.data_csv_path)
        required_columns = ['image_path', 'class_name']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")
        
        # Check if image paths exist
        missing_files = []
        for idx, row in df.iterrows():
            img_path = row['image_path']
            if not os.path.exists(img_path):
                missing_files.append(img_path)
                if len(missing_files) >= 5:  # Limit checking to avoid too many checks
                    break
        
        if missing_files:
            raise FileNotFoundError(f"Some image files are missing. First few: {missing_files[:5]}")
            
        return True
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")
    except pd.errors.ParserError:
        raise ValueError("The CSV file is not properly formatted")

class ImageDataset(Dataset):
    """Custom Dataset for loading images with their labels."""
    
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None, is_training: bool = True):
        self.df = df
        
        # Use appropriate transforms based on dataset purpose
        if transform is not None:
            self.transform = transform
        elif is_training:
            self.transform = DataAugmentation.get_train_transforms()
        else:
            self.transform = DataAugmentation.get_valid_transforms()
        
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

def create_data_loaders(split_file: Optional[str] = None, save_dir: Optional[str] = None,
                       use_augmentation: bool = True
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from the CSV file.
    
    Args:
        split_file: Optional path to a JSON file containing split indices
        save_dir: Optional directory to save split indices if not loading from file
        use_augmentation: Whether to use data augmentation for training
    
    Returns:
        Tuple containing train, validation, and test DataLoader objects
    """
    # Validate the CSV file
    validate_data_csv()
    
    # Read the CSV file
    df = pd.read_csv(config.data_csv_path)
    
    # Create transforms
    train_transform = DataAugmentation.get_train_transforms() if use_augmentation else DataAugmentation.get_valid_transforms()
    valid_transform = DataAugmentation.get_valid_transforms()
    test_transform = DataAugmentation.get_test_transforms()
    
    # Create datasets
    train_dataset = ImageDataset(df, transform=train_transform, is_training=True)
    valid_dataset = ImageDataset(df, transform=valid_transform, is_training=False)
    test_dataset = ImageDataset(df, transform=test_transform, is_training=False)
    
    if split_file and os.path.exists(split_file):
        # Load existing splits
        train_idx, val_idx, test_idx = load_split_indices(split_file)
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(valid_dataset, val_idx)
        test_dataset = Subset(test_dataset, test_idx)
    else:
        # Create new splits
        total_size = len(train_dataset)
        train_size = int(total_size * config.train_split)
        val_size = int(total_size * config.val_split)
        test_size = total_size - train_size - val_size
        
        # Split the dataset with fixed seed
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            train_dataset, 
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
    
    # Determine optimal number of workers based on CPU cores
    num_workers = min(os.cpu_count() or 4, config.num_workers * 2)
    
    # Create optimized data loaders
    # Use more workers and persistent workers for training
    # Use pinned memory for faster CPU->GPU transfers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # Drop last incomplete batch for better optimization
    )
    
    # Use fewer workers for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Can use larger batches for validation
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,  # Can use larger batches for testing
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
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