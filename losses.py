import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multilabel classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
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


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multilabel classification."""
    
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, 
                 clip: float = 0.05, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric focusing
        if self.gamma_neg > 0:
            xs_neg = (xs_neg ** self.gamma_neg).clamp(max=self.clip)
        if self.gamma_pos > 0:
            xs_pos = (xs_pos ** self.gamma_pos).clamp(max=self.clip)
            
        # Calculate loss
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        
        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for handling class imbalance."""
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.pos_weight, reduction=self.reduction
        )


def calculate_class_weights(labels: np.ndarray, method: str = 'inverse_freq') -> torch.Tensor:
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


def get_loss_function(
        loss_type: str = 'focal',
        labels: Optional[np.ndarray] = None,
        device: str = 'cuda',
        class_weight_method: str = 'inverse_freq'
) -> nn.Module:
    """Get the appropriate loss function for multilabel classification.
    
    Args:
        loss_type: Type of loss function ('focal', 'asymmetric', 'weighted_bce', 'bce')
        labels: Training labels for calculating class weights
        device: Device to place weights on
        class_weight_method: Method for calculating class weights
    
    Returns:
        Loss function module
    """
    
    if loss_type == 'focal':
        return FocalLoss(alpha=1.0, gamma=2.0)
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    
    elif loss_type == 'weighted_bce':
        if labels is not None:
            pos_weight = calculate_class_weights(labels, method=class_weight_method)
            pos_weight = pos_weight.to(device)
            return WeightedBCELoss(pos_weight=pos_weight)
        else:
            return WeightedBCELoss()
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
