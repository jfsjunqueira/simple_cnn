import torch
from torchvision import transforms
from typing import Callable, Optional, Dict, Any
from .config import config

class DataAugmentation:
    """
    Class for handling data augmentation strategies for CNN training.
    
    This provides various augmentation pipelines that can be used for different
    purposes like training, validation, and testing.
    """
    
    @staticmethod
    def get_train_transforms(
        height: int = config.image_height,
        width: int = config.image_width,
        mean: list = [0.485, 0.456, 0.406],  # ImageNet mean
        std: list = [0.229, 0.224, 0.225],   # ImageNet std
        **kwargs: Any
    ) -> Callable:
        """
        Get data augmentation pipeline for training.
        
        Args:
            height: Target image height
            width: Target image width
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            **kwargs: Additional parameters for customization
            
        Returns:
            Composition of transforms for training augmentation
        """
        return transforms.Compose([
            transforms.RandomResizedCrop((height, width), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    @staticmethod
    def get_valid_transforms(
        height: int = config.image_height,
        width: int = config.image_width,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        **kwargs: Any
    ) -> Callable:
        """
        Get transforms for validation/testing.
        
        Args:
            height: Target image height
            width: Target image width
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            **kwargs: Additional parameters for customization
            
        Returns:
            Composition of transforms for validation/testing
        """
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    @staticmethod
    def get_test_transforms(
        height: int = config.image_height,
        width: int = config.image_width,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        **kwargs: Any
    ) -> Callable:
        """Alias for get_valid_transforms (no augmentation for testing)"""
        return DataAugmentation.get_valid_transforms(
            height=height, width=width, mean=mean, std=std, **kwargs
        )
    
    @staticmethod
    def get_strong_augmentation(
        height: int = config.image_height,
        width: int = config.image_width,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        **kwargs: Any
    ) -> Callable:
        """
        Get a stronger augmentation pipeline for training with limited data.
        
        Args:
            height: Target image height
            width: Target image width
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            **kwargs: Additional parameters for customization
            
        Returns:
            Composition of transforms with stronger augmentation
        """
        return transforms.Compose([
            transforms.RandomResizedCrop((height, width), scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]) 