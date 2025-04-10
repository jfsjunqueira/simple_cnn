import argparse
import os
import torch
from torch.serialization import safe_globals
from src.model import SimpleCNN
from src.metrics import ModelEvaluator
from src.image_utils import create_data_loaders
from src.config import config, CNNConfig

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

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN model')
    parser.add_argument('model_path', type=str, help='Path to the saved model checkpoint')
    parser.add_argument('--split-file', type=str, help='Path to dataset split file (optional)')
    parser.add_argument('--output-dir', type=str, help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_dir = os.path.dirname(args.model_path)
        output_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = load_model(args.model_path, device)
    
    # Create data loader using same split if available
    print("\nCreating data loaders...")
    _, _, test_loader = create_data_loaders(split_file=args.split_file)
    
    # Create evaluator and compute metrics
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.compute_metrics(test_loader, save_dir=output_dir)
    evaluator.save_metrics(metrics, output_dir)
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"Test Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()