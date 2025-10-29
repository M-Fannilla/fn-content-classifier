import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import DEVICE


class WeightedBCELoss(nn.Module):
    def __init__(self, class_freq, reduction="mean", power=1.0):
        super().__init__()

        # Inverse frequency → higher weight for rare positives
        pos_weight = 1.0 / (class_freq + 1e-8)

        # Mild scaling — power < 1 keeps weights controlled
        pos_weight = pos_weight**power
        pos_weight = pos_weight / pos_weight.mean()

        self.register_buffer(
            "pos_weight", torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)
        )
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )