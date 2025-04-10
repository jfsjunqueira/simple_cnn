# Simple CNN

A simple CNN implementation for image classification using PyTorch.

## Setup

### Requirements

- Python 3.10+
- PyTorch with CUDA support (optional, but recommended for GPU acceleration)

### Installation

0. Install [uv](https://github.com/astral-sh/uv)

1. Clone the repository:
```bash
git clone https://github.com/username/simple_cnn.git
cd simple_cnn
```

2. Set up a virtual environment:
```bash
uv sync
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

### Preparing Data

The training script expects a CSV file with the following format:
```
image_path,class_name
/path/to/image1.jpg,cat
/path/to/image2.jpg,dog
...
```

The default location for this file is `data/img_labels.csv`. You can modify this path in `src/config.py`.

### Training a Model

To train a model with default settings:
```bash
python -m src.main
```

To resume training from a checkpoint:
```bash
python -m src.main --resume path/to/checkpoint.pth
```

To specify the number of epochs:
```bash
python -m src.main --epochs 100
```

To force using CPU even when GPU is available:
```bash
python -m src.main --cpu
```

### Model Output

After training, the model and related artifacts will be saved in a timestamped directory under `models/`:
- `model.pth`: The trained model
- `checkpoints/`: Directory containing training checkpoints
- `metrics/`: Directory containing evaluation metrics and visualizations

### Performance Notes

Performance varies significantly across hardware platforms:
- NVIDIA GPUs with CUDA typically achieve the highest throughput (15-20 batches/sec on RTX 3070)
- Apple Silicon (M1/M2/M3) using MPS backend is slower (around 5 batches/sec)
- CPU training is much slower and not recommended for large datasets

These differences are expected and related to hardware capabilities and backend optimizations.

## Project Structure

- `src/`: Source code
  - `model.py`: CNN architecture definition
  - `trainer.py`: Training code
  - `image_utils.py`: Image loading and processing utilities
  - `data_augmentation.py`: Data augmentation techniques
  - `metrics.py`: Evaluation metrics
  - `config.py`: Configuration options
  - `main.py`: Main training script
- `notebooks/`: Jupyter notebooks for exploration
- `scripts/`: Utility scripts

## Configuration

Model and training parameters can be modified in `src/config.py`.

## License

[MIT License](LICENSE)
