import argparse
import os
import torch
from torch.serialization import safe_globals
import pandas as pd
from src.model import SimpleCNN
from src.image_utils import load_and_transform_image
from src.config import CNNConfig
import numpy as np

def load_model(model_path: str, device: torch.device) -> SimpleCNN:
    """Load model from checkpoint and move to correct device"""
    # Allow CNNConfig class to be loaded from the checkpoint
    with safe_globals([CNNConfig]):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model and load state
    model = SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def get_class_mapping(csv_path: str) -> dict:
    """Get mapping of class indices to class names from the dataset"""
    df = pd.read_csv(csv_path)
    unique_classes = sorted(df['class_name'].unique())
    return {idx: name for idx, name in enumerate(unique_classes)}

def predict_image(model: torch.nn.Module, image_path: str, device: torch.device) -> torch.Tensor:
    """Run inference on a single image and return probabilities"""
    # Load and preprocess the image
    image_tensor = load_and_transform_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    return probabilities.cpu()

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('model_path', type=str, help='Path to the saved model checkpoint')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--csv-path', type=str, 
                       default='data/img_labels.csv',
                       help='Path to the CSV file with class labels')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Show top K predictions (default: 3)')
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = load_model(args.model_path, device)
    
    # Get class names
    class_mapping = get_class_mapping(args.csv_path)
    
    # Run prediction
    print(f"\nProcessing image: {args.image_path}")
    probabilities = predict_image(model, args.image_path, device)
    
    # Sort probabilities and get indices
    probs, indices = torch.sort(probabilities, descending=True)
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"{'Class':<20} {'Probability':>10}")
    print("-" * 50)
    
    # Show all probabilities, highlight top k
    for i, (prob, idx) in enumerate(zip(probs, indices)):
        class_name = class_mapping[idx.item()]
        probability = prob.item() * 100
        
        if i < args.top_k:
            print(f"{class_name:<20} {probability:>10.2f}%  ←")
        else:
            print(f"{class_name:<20} {probability:>10.2f}%")
    
    # Print most likely prediction
    top_class = class_mapping[indices[0].item()]
    top_prob = probs[0].item() * 100
    print("\nMost likely class:", f"{top_class} ({top_prob:.2f}%)")

if __name__ == "__main__":
    main()