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
        config: Config
) -> ConvNeXtV2MultilabelClassifier:
    """Setup model for finetuning (freeze backbone, train classifier only)."""

    for name, param in model.backbone.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            param.requires_grad = False

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
