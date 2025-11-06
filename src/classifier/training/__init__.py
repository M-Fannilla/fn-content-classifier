from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory for Torch models
TORCH_MODELS_DIR = Path("./models/torch")
TORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training-related directories
DATASETS_DIR = Path('./fn-content-dataset/images')