import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import (
    DATASETS_DIR,
    DEVICE,
    ONNX_DIR,
)


@dataclass
class TrainConfig:
    # Dataset settings
    model_type: str = 'action'
    train_size_perc: float = 0.8
    test_size_perc: float = 0.2

    label_dataframe: str = f"{DATASETS_DIR}/{model_type}_labels.csv"

    # Training settings
    seed: int = 42
    batch_size: int = 32
    num_workers: int = os.cpu_count() // 2
    img_size: int = 384
    model_name: str = 'convnextv2_tiny'

    # Finetuning settings
    weight_decay: float = 1e-4
    learning_rate: float = 1.25E-05
    epochs: int = 15
    bce_power: float = 0.5
    tau_logit_adjust: float = 0.5

    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = 'fn-content-classifier'
    wandb_entity: str = 'miloszbertman'  # Set to your wandb username/team
    wandb_run_name: str = None  # Will be auto-generated if None
    wandb_tags: list = None

    # Model settings
    reduce_metric: str = 'pr_auc_macro'
    early_stop_metric: str = 'pr_auc_macro'
    best_model_metric: str = "pr_auc_macro"

    cosine_annealing_min: float = 0.1

    pretrained: bool = True

    @property
    def model_catalog(self) -> dict[str, int]:
        return {
            'convnextv2_nano': self.img_size,
            'convnextv2_tiny': self.img_size,
            'convnextv2_base': self.img_size,
            'convnextv2_large': self.img_size,
            'convnextv2_huge': self.img_size,
        }


    def info(self) -> None:
        print("Configuration:")
        for field in self.valid_params():
            print(f"  {field}: {getattr(self, field)}")

    def wandb_config(self):
        if not self.use_wandb:
            print("Wandb is disabled.")
            return

        print("Wandb Configuration:")
        print(f"  Enabled: {self.use_wandb}")
        print(f"  Project: {self.wandb_project}")
        print(f"  Entity: {self.wandb_entity}")
        print(f"  Tags: {self.wandb_tags}")

    @classmethod
    def valid_params(cls) -> list[str]:
        return [field.name for field in fields(cls)]


@dataclass
class TorchModelConfig:
    model_type: str
    model_state_dict: dict
    labels: list[str]
    image_size: int
    threshold: np.ndarray

    def save_torch_model(self, save_path: str) -> None:
        torch.save(self.__dict__, save_path)
        print(f"Saved PyTorch model to {save_path}")

    @classmethod
    def load_torch_model(cls, model_path: str) -> 'TorchModelConfig':
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

        config_args = [str(f.name) for f in fields(cls)]
        return TorchModelConfig(
            **{
                field: checkpoint.get(field) for field in config_args
            }
        )

    def save_as_onnx_config(self, epoch: int | str) -> None:
        save_path = OnnxModelConfig(
            model_type=self.model_type,
            labels=self.labels,
            image_size=self.image_size,
            threshold=self.threshold,
        ).save_config(epoch=epoch)
        print(f"[✓] Saved config for model: {save_path}")

    def export_to_onnx(self, model: nn.Module) -> None:
        model.eval().to(DEVICE)

        image_shape = (1, 3, self.image_size, self.image_size)
        example_input = torch.randn(*image_shape, device=DEVICE)
        onnx_path = ONNX_DIR / f"{self.model_type}.onnx"

        torch.onnx.export(
            model,
            example_input,
            f=onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["x"],
            output_names=["output"],
            dynamo=False,
            dynamic_shapes={
                "x": {0: "batch"},
            },
        )
        print(f"[✓] Saved ONNX model: {onnx_path}")
        return onnx_path


@dataclass
class OnnxModelConfig:
    model_type: str
    labels: list[str]
    image_size: int
    threshold: np.ndarray | list[float]

    def save_config(self, epoch: int | str) -> Path:
        save_path = ONNX_DIR / f"{str(epoch)}_{self.model_type}.json"

        if isinstance(self.threshold, np.ndarray):
            self.threshold = self.threshold.tolist()

        self.threshold = [round(t, 5) for t in self.threshold]

        with open(save_path, 'w') as f:
            json.dump(self.__dict__, f)

        print(f"[✓] Saved OnnxModelConfig to {save_path}")
        return save_path

    @staticmethod
    def load_config(model_type: str) -> 'OnnxModelConfig':
        load_path = ONNX_DIR / f"{model_type}.json"

        with open(load_path, 'r') as f:
            config_dict = json.load(f)

        config = OnnxModelConfig(**config_dict)
        config.threshold = np.array(config.threshold)
        print(f"[✓] Loaded OnnxModelConfig from {load_path}")

        return config