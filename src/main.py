import torch
import os
import time
from datetime import datetime, timedelta
from .config import config
from .model import SimpleCNN
from .trainer import CNNTrainer
from .image_utils import create_data_loaders
from .metrics import ModelEvaluator

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", timestamp)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory created at: {model_dir}")
    
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
    print(f"  Epochs:        {config.epochs}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Initial LR:    {config.initial_lr}")
    print(f"  Scheduler:     {config.lr_scheduler_type}")
    train_start = time.time()
    
    history = trainer.train(train_loader, val_loader)
    
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
    print(f"Best val acc:      {max(history['val_accuracy']) * 100:.2f}%")
    print(f"Results saved in:  {model_dir}")

if __name__ == "__main__":
    main()