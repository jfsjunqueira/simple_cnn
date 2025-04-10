import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = config.channels
        
        for out_channels in config.conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
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
        
        for out_features in config.fc_layers:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))
            in_features = out_features
        
        # Final classification layer
        self.classifier = nn.Linear(config.fc_layers[-1], config.num_classes)
        
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