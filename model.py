import torch
import torch.nn as nn
import timm
from config import Config


class ConvNeXtV2MultilabelClassifier(nn.Module):
    """ConvNeXt V2 model for multilabel classification."""

    def __init__(
            self,
            model_name: str,
            num_classes: int,
            pretrained: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


def create_model(config: Config, num_classes: int) -> ConvNeXtV2MultilabelClassifier:
    """Create and configure the model."""
    return ConvNeXtV2MultilabelClassifier(
        model_name=config.model_name,
        num_classes=num_classes,
        pretrained=True,
    )


def setup_model_for_training(
        model: ConvNeXtV2MultilabelClassifier,
        device: torch.device
) -> ConvNeXtV2MultilabelClassifier:
    """Setup model for finetuning (freeze backbone, train classifier only)."""

    for name, param in model.backbone.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            param.requires_grad = False

    # Count parameters
    param_counts = count_parameters(model)
    print(f"Model created successfully!")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters: {param_counts['frozen']:,}")
    print(f"  Training mode: Finetuning (backbone frozen, classifier trainable)")

    # Move model to device
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
