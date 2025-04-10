import torch
import os
import time
import argparse
from datetime import datetime, timedelta
from .config import config
from .model import SimpleCNN
from .trainer import CNNTrainer
from .image_utils import create_data_loaders
from .metrics import ModelEvaluator

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Simple CNN model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU even if GPU is available')
    return parser.parse_args()

def get_device(force_cpu=False):
    """
    Determine the best available device for training.
    
    Args:
        force_cpu: If True, will use CPU regardless of GPU availability
        
    Returns:
        torch.device: The device to use for training
    """
    if force_cpu:
        return torch.device('cpu')
        
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # For Apple Silicon (M1/M2/M3)
        return torch.device('mps')
    else:
        return torch.device('cpu')

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up device
    device = get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", timestamp)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory created at: {model_dir}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Create data loaders and save splits
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(save_dir=model_dir)
    num_train = len(train_loader.dataset)
    num_val = len(val_loader.dataset)
    num_test = len(test_loader.dataset)
    print(f"Dataset splits:")
    print(f"  Training:   {num_train} images")
    print(f"  Validation: {num_val} images")
    print(f"  Test:       {num_test} images")
    
    # Initialize model and move to device
    print("\nInitializing model...")
    model = SimpleCNN()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")
    
    # Create trainer and evaluator
    trainer = CNNTrainer(model)
    evaluator = ModelEvaluator(model, device)
    
    # Train the model
    print("\nStarting training...")
    print(f"Training configuration:")
    print(f"  Epochs:        {args.epochs or config.epochs}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Initial LR:    {config.initial_lr}")
    print(f"  Scheduler:     {config.lr_scheduler_type}")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
    
    train_start = time.time()
    
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume
    )
    
    train_time = time.time() - train_start
    print(f"\nTraining completed in {format_time(train_time)}")
    
    # Save the trained model
    model_path = os.path.join(model_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': history
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    metrics = evaluator.compute_metrics(
        test_loader,
        save_dir=os.path.join(model_dir, "metrics")
    )
    evaluator.save_metrics(metrics, os.path.join(model_dir, "metrics"))
    
    # Print final summary
    total_time = time.time() - start_time
    print("\nFinal Summary:")
    print(f"Total time:        {format_time(total_time)}")
    print(f"Training time:     {format_time(train_time)}")
    print(f"Final test acc:    {metrics['overall_accuracy']:.2f}%")
    
    # Handle case where validation didn't occur (e.g. if interrupted)
    if history['val_accuracy'] and len(history['val_accuracy']) > 0:
        best_val_acc = max(history['val_accuracy']) * 100
        print(f"Best val acc:      {best_val_acc:.2f}%")
    else:
        print("Best val acc:      N/A (no validation performed)")
    
    print(f"Results saved in:  {model_dir}")

if __name__ == "__main__":
    main()