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


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate class weights for handling imbalance."""
    # Calculate positive class frequency
    pos_freq = labels.mean(axis=0)
    
    # Calculate weights (inverse frequency)
    weights = 1.0 / (pos_freq + 1e-8)
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return torch.FloatTensor(weights)


def get_loss_function(loss_type: str = 'focal', 
                     labels: Optional[np.ndarray] = None,
                     device: str = 'cuda') -> nn.Module:
    """Get the appropriate loss function for multilabel classification."""
    
    if loss_type == 'focal':
        return FocalLoss(alpha=1.0, gamma=2.0)
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    
    elif loss_type == 'weighted_bce':
        if labels is not None:
            pos_weight = calculate_class_weights(labels)
            pos_weight = pos_weight.to(device)
            return WeightedBCELoss(pos_weight=pos_weight)
        else:
            return WeightedBCELoss()
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
