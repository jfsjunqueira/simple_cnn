import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from .config import config

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    def evaluate(self, data_loader: DataLoader) -> Tuple[List[int], List[int], List[float]]:
        """
        Evaluate model on a dataset and return predictions, true labels and probabilities.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return all_preds, all_labels, all_probs
    
    def compute_metrics(self, data_loader: DataLoader, save_dir: str = None) -> Dict:
        """
        Compute and optionally save various metrics including:
        - Per-class precision, recall, F1-score
        - Confusion matrix
        - Overall accuracy
        """
        predictions, true_labels, probabilities = self.evaluate(data_loader)
        
        # Get class names from the data loader's dataset
        class_names = [str(i) for i in range(config.num_classes)]
        
        # Generate classification report
        report = classification_report(true_labels, predictions, 
                                    target_names=class_names, 
                                    output_dict=True)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot and save confusion matrix if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close()
        
        # Compute per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        metrics = {
            'classification_report': report,
            'confusion_matrix': cm,
            'per_class_accuracy': {
                class_name: acc for class_name, acc in zip(class_names, per_class_accuracy)
            },
            'overall_accuracy': report['accuracy']
        }
        
        return metrics
    
    def save_metrics(self, metrics: Dict, save_dir: str):
        """Save metrics to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save classification report
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            for class_name, metrics_dict in metrics['classification_report'].items():
                if isinstance(metrics_dict, dict):
                    f.write(f"\nClass: {class_name}\n")
                    for metric_name, value in metrics_dict.items():
                        f.write(f"{metric_name}: {value:.4f}\n")
                else:
                    f.write(f"\n{class_name}: {metrics_dict:.4f}\n")
        
        # Save per-class accuracy
        with open(os.path.join(save_dir, 'per_class_accuracy.txt'), 'w') as f:
            for class_name, accuracy in metrics['per_class_accuracy'].items():
                f.write(f"{class_name}: {accuracy:.4f}\n")
            f.write(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}\n")