import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize convolutional layers with efficient blocks
        self.conv_layers = nn.ModuleList()
        in_channels = config.channels
        
        for out_channels in config.conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),  # inplace operation saves memory
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels
        
        # Calculate the size of flattened features after convolutions
        feature_size = (config.image_height // (2 ** len(config.conv_channels))) * \
                      (config.image_width // (2 ** len(config.conv_channels))) * \
                      config.conv_channels[-1]
        
        # Initialize fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = feature_size
        
        for i, out_features in enumerate(config.fc_layers):
            # Use lower dropout in earlier layers
            dropout_rate = 0.3 if i == 0 else 0.5
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),  # Add BN to FC layers
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ))
            in_features = out_features
        
        # Final classification layer
        self.classifier = nn.Linear(config.fc_layers[-1], config.num_classes)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize the weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            
        # Apply final classification
        return self.classifier(x)