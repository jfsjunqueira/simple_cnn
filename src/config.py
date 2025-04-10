from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CNNConfig:
    # Image dimensions
    image_height: int = 224
    image_width: int = 224
    channels: int = 3  # RGB images
    
    # CNN architecture
    conv_channels: list[int] = (32, 64, 128)  # Number of filters in each conv layer
    fc_layers: list[int] = (512, 256)  # Fully connected layer dimensions
    
    # Class configuration
    class_labels: List[str] = ('class_0', 'class_1', 'class_2', 'class_3', 'class_4',
                              'class_5', 'class_6', 'class_7', 'class_8', 'class_9')
    num_classes: int = 10  # Number of output classes
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 50
    
    # Learning rate configuration
    initial_lr: float = 0.001
    min_lr: float = 1e-6
    lr_scheduler_type: str = 'cosine'  # Options: 'cosine', 'step', 'exponential'
    lr_scheduler_params: dict = None

    # Dataset configuration
    data_csv_path: str = r"C:\Users\jfsju\Projetos\simple_cnn\data\img_labels.csv"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    
    def __post_init__(self):
        self.num_classes = len(self.class_labels)
        if self.lr_scheduler_params is None:
            self.lr_scheduler_params = {
                'cosine': {'T_max': self.epochs},
                'step': {'step_size': 30, 'gamma': 0.1},
                'exponential': {'gamma': 0.95}
            }[self.lr_scheduler_type]
        
        # Validate split ratios
        total_split = self.train_split + self.val_split + self.test_split
        if not 0.99 <= total_split <= 1.01:  # Allow for small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
    
config = CNNConfig()