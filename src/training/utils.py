import os
import torch
from sklearn.metrics import f1_score
import numpy as np

def find_best_thresholds(y_true, y_prob, step=0.01):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = (y_true > 0.5).astype(np.int32)
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    _, C = y_true.shape
    thresholds = np.zeros(C, dtype=np.float32)
    for feature_idx in range(C):
        best_thr, best_f1 = 0.0, 0.0
        probs = y_prob[:, feature_idx]

        for thr in np.arange(0.05, 0.95, step):
            f1 = f1_score(
                y_true[:, feature_idx], (probs > thr).astype(np.int32),
                zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds[feature_idx] = best_thr

    print("Thresholds:\n", [round(float(t), 2) for t in thresholds])
    return thresholds


def compute_class_frequency(df)-> np.ndarray:
    pos_counts = df.sum(axis=0).values  # shape (C,)
    total_samples = len(df)
    pos_freq = pos_counts / total_samples
    return np.clip(pos_freq, 1e-12, 1.0)  # avoid divide-by-zero issues


def print_system_info():
    """Print system information and configuration."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 60)