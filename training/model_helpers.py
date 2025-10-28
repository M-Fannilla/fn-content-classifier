import numpy as np
import torch
import torch.nn as nn
import timm
from training.config import Config
from model import ClassifierModel


def create_model(config: Config, num_classes: int) -> nn.Module:
    """Create and configure the model."""
    return timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            in_chans=3,
            num_classes=num_classes,
    )

def setup_model_for_training(
        config: Config,
        model: nn.Module,
        device: torch.device,
        class_freq: np.ndarray,
) -> ClassifierModel:
    """Setup model for finetuning (freeze backbone, train classifier only)."""

    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            param.requires_grad = True

    # Count parameters
    param_counts = count_parameters(model)
    print("Model created successfully!")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters: {param_counts['frozen']:,}")
    print("  Training mode: Finetuning (backbone frozen, classifier trainable)")

    model = ClassifierModel(
        backbone=model,
        class_freq=class_freq,
        tau=config.tau_logit_adjust
    )
    print(f"  Logit adjustment applied with tau={config.tau_logit_adjust}")

    model = model.to(device)
    print(f"  Model moved to: {device}")
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }
