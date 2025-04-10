import torch
from PIL import Image
from torchvision import transforms
from .config import config

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