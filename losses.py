import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multilabel classification."""

    def __init__(
            self,
            alpha: float = 1.0,
            gamma: float = 2.0,
            reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedBCELoss(nn.Module):
    def __init__(self, class_freq, device, reduction="mean", power=1.0):
        super().__init__()

        # Inverse frequency → higher weight for rare positives
        pos_weight = 1.0 / (class_freq + 1e-8)
        # Mild scaling — power < 1 keeps weights controlled
        pos_weight = pos_weight**power
        pos_weight = pos_weight / pos_weight.mean()

        self.register_buffer(
            "pos_weight", torch.tensor(pos_weight, dtype=torch.float32).to(device)
        )
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


def calculate_class_weights(
        labels: np.ndarray, method: str = "inverse_freq"
) -> torch.Tensor:
    """Calculate class weights for handling imbalance.

    Args:
        labels: Binary labels array of shape (n_samples, n_classes)
        method: Method to calculate weights ('inverse_freq', 'balanced', 'sqrt_inverse_freq')

    Returns:
        Class weights tensor of shape (n_classes,)
    """
    # Calculate positive class frequency
    pos_freq = labels.mean(axis=0)
    n_classes = len(pos_freq)

    if method == 'inverse_freq':
        # Inverse frequency weighting
        weights = 1.0 / (pos_freq + 1e-8)
        # Normalize weights
        weights = weights / weights.sum() * n_classes

    elif method == 'balanced':
        # Balanced weighting (sklearn style)
        n_samples = len(labels)
        n_pos = labels.sum(axis=0)
        n_neg = n_samples - n_pos

        # Avoid division by zero
        weights = n_samples / (2.0 * np.maximum(n_pos, 1e-8))

    elif method == 'sqrt_inverse_freq':
        # Square root of inverse frequency
        weights = 1.0 / np.sqrt(pos_freq + 1e-8)
        # Normalize weights
        weights = weights / weights.sum() * n_classes

    else:
        raise ValueError(f"Unknown class weight method: {method}")

    return torch.FloatTensor(weights)


def print_class_weights(
        labels: np.ndarray,
        class_names: List[str],
        method: str = 'inverse_freq'
) -> torch.Tensor:
    """Calculate and print class weights with detailed information.

    Args:
        labels: Binary labels array of shape (n_samples, n_classes)
        class_names: List of class names
        method: Method to calculate weights

    Returns:
        Class weights tensor
    """
    pos_freq = labels.mean(axis=0)
    weights = calculate_class_weights(labels, method)

    print(f"\nClass Imbalance Analysis (Method: {method}):")
    print("=" * 60)
    print(f"{'Class':<20} {'Pos Freq':<10} {'Count':<8} {'Weight':<10}")
    print("-" * 60)

    for i, (class_name, freq, weight) in enumerate(zip(class_names, pos_freq, weights)):
        count = int(freq * len(labels))
        print(f"{class_name:<20} {freq:<10.4f} {count:<8} {weight:<10.4f}")

    print("-" * 60)
    print(f"Total samples: {len(labels)}")
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Weight std: {weights.std():.4f}")
    print("=" * 60)

    return weights