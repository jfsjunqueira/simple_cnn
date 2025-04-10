import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from .config import config
from .model import SimpleCNN

class CNNTrainer:
    def __init__(self, model: Optional[SimpleCNN] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model and return average loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / num_batches, correct / total

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: Optional[int] = None) -> dict:
        """
        Train the model for the specified number of epochs.
        Returns a dictionary containing training history.
        """
        epochs = epochs or config.epochs
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }

        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            self.scheduler.step()
            
        return history

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, list[str]]:
        """
        Make predictions on the input tensor and return both raw probabilities
        and predicted class labels.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            predicted_indices = probs.argmax(dim=1)
            predicted_labels = [config.class_labels[idx] for idx in predicted_indices]
            
        return probs, predicted_labels