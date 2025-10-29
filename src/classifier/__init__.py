import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory for Torch models
TORCH_MODELS_DIR = Path("./models/torch")
TORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Directory for ONNX models
ONNX_DIR = Path("./models/onnx")
ONNX_DIR.mkdir(parents=True, exist_ok=True)

# Training-related directories
DATASETS_DIR = Path('./fn-content-dataset/compiled')