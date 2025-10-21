import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any
from config import Config


class ConvNeXtV2MultilabelClassifier(nn.Module):
    """ConvNeXt V2 model for multilabel classification."""
    
    def __init__(self, 
                 model_name: str, 
                 num_classes: int, 
                 pretrained: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove the original classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_model(config: Config, num_classes: int) -> ConvNeXtV2MultilabelClassifier:
    """Create and configure the model."""
    model = ConvNeXtV2MultilabelClassifier(
        model_name=config.model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.2
    )
    
    return model


def setup_model_for_training(model: ConvNeXtV2MultilabelClassifier, 
                           config: Config) -> ConvNeXtV2MultilabelClassifier:
    """Setup model for finetuning (freeze backbone, train classifier only)."""
    
    # Freeze backbone for finetuning
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Only train the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }
