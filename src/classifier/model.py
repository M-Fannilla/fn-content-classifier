import numpy as np
import torch
import torch.nn as nn

class LogitAdjustment(nn.Module):
    def __init__(self, class_freq: torch.Tensor, tau: float=1.0):
        super().__init__()
        self.register_buffer("log_adj", tau * torch.log(class_freq))

    def forward(self, x):
        return x - self.log_adj

class ClassifierModel(nn.Module):
    """Custom classifier model wrapping a timm backbone."""

    def __init__(self, backbone: nn.Module, class_freq: np.ndarray, tau: float):
        super().__init__()
        self.backbone = backbone
        self.tau = tau
        self.logit_adjust = LogitAdjustment(class_freq=torch.tensor(class_freq), tau=tau)

    def forward(self, x):
        logits = self.backbone(x)
        return self.logit_adjust(logits)