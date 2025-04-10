import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import time
import signal
import sys
import os
from .config import config
from .model import SimpleCNN
from .gpu_utils import clean_gpu_memory

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, *args):
        print("\n\nReceived interrupt signal. Will stop after this epoch...")
        self.kill_now = True

def get_device():
    """
    Determine the best available device for training.
    
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # For Apple Silicon (M1/M2/M3)
        return torch.device('mps')
    else:
        return torch.device('cpu')

class CNNTrainer:
    def __init__(self, model: Optional[SimpleCNN] = None):
        self.device = get_device()
        self.model = model if model is not None else SimpleCNN()
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.initial_lr)
        
        # Setup learning rate scheduler
        if config.lr_scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config.lr_scheduler_params['T_max'],
                eta_min=config.min_lr
            )
        elif config.lr_scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_scheduler_params['step_size'],
                gamma=config.lr_scheduler_params['gamma']
            )
        else:  # exponential
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=config.lr_scheduler_params['gamma']
            )

    def cleanup(self):
        """Clean up resources and GPU memory"""
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        if hasattr(self, 'criterion'):
            del self.criterion
        clean_gpu_memory()

    def train_epoch(self, train_loader: DataLoader, epoch: int, killer: GracefulKiller) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}',
                   leave=False, unit='batch')
        
        # Track metrics for this epoch
        correct = 0
        total = 0
        start_time = time.time()

        try:
            for inputs, targets in pbar:
                if killer.kill_now:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_loss = total_loss / (pbar.n + 1)
                current_acc = 100. * correct / total
                current_lr = self.optimizer.param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%',
                    'lr': f'{current_lr:.6f}'
                })

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            self.cleanup()
            raise e

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / num_batches
        epoch_acc = 100. * correct / total
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {epoch_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}\n')
        
        return epoch_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model and return average loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for validation
        pbar = tqdm(val_loader, desc='Validating', leave=False, unit='batch')

        try:
            with torch.no_grad():
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Update progress bar
                    current_loss = total_loss / (pbar.n + 1)
                    current_acc = 100. * correct / total
                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{current_acc:.2f}%'
                    })
        except Exception as e:
            print(f"\nError during validation: {str(e)}")
            self.cleanup()
            raise e

        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        
        print(f'\nValidation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'Accuracy: {val_acc * 100:.2f}%\n')
        
        return val_loss, val_acc

    def save_checkpoint(self, checkpoint_dir: str, epoch: int, val_acc: float,
                      history: Dict, is_best: bool = False) -> str:
        """
        Save a checkpoint of the model's state to resume training later.
        
        Args:
            checkpoint_dir: Directory to save the checkpoint
            epoch: Current epoch number
            val_acc: Current validation accuracy
            history: Training history dictionary
            is_best: Whether this model has the best validation accuracy so far
            
        Returns:
            Path to the saved checkpoint file
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': history,
            'val_acc': val_acc
        }
        
        torch.save(checkpoint, checkpoint_file)
        
        # Also save as best model if it's the best so far
        if is_best:
            best_file = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_file)
            
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, Dict]:
        """
        Load a model checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Tuple of (epoch to start from, training history)
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore optimizer state to the right device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        
        return checkpoint['epoch'], checkpoint['history']

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: Optional[int] = None, checkpoint_dir: Optional[str] = None,
              resume_from: Optional[str] = None) -> dict:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train for (defaults to config.epochs)
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to a checkpoint file to resume training from
        """
        epochs = epochs or config.epochs
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
            'completed_epochs': 0
        }
        
        start_epoch = 0
        best_val_acc = 0.0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            start_epoch, history = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch + 1}")
            # Update start_epoch to continue from the next epoch
            start_epoch += 1
        
        killer = GracefulKiller()
        
        try:
            # Create progress bar for epochs
            pbar = tqdm(range(start_epoch, epochs), desc='Training Progress', unit='epoch')
            
            for epoch in pbar:
                # Training phase
                train_loss = self.train_epoch(train_loader, epoch, killer)
                history['train_loss'].append(train_loss)
                history['completed_epochs'] = epoch + 1
                
                if killer.kill_now:
                    print("\nTraining interrupted. Saving progress...")
                    break
                
                # Validation phase
                if val_loader:
                    val_loss, val_acc = self.validate(val_loader)
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_acc)
                    
                    # Save checkpoint
                    if checkpoint_dir:
                        is_best = val_acc > best_val_acc
                        if is_best:
                            best_val_acc = val_acc
                        
                        self.save_checkpoint(
                            checkpoint_dir,
                            epoch,
                            val_acc,
                            history,
                            is_best
                        )
                
                # Update learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                self.scheduler.step()
        
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            self.cleanup()
            raise e
        
        return history

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, list[str]]:
        """
        Make predictions on the input tensor and return both raw probabilities
        and predicted class labels.
        """
        self.model.eval()
        try:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                predicted_indices = probs.argmax(dim=1)
                predicted_labels = [config.class_labels[idx] for idx in predicted_indices]
                
            return probs, predicted_labels
        finally:
            inputs = inputs.cpu()
            if hasattr(outputs, 'cpu'):
                outputs = outputs.cpu()
            if hasattr(probs, 'cpu'):
                probs = probs.cpu()