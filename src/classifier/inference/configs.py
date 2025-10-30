import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import ONNX_DIR
from pydantic_settings import BaseSettings


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



class InferenceConfig(BaseSettings):
    IMAGE_PROCESSING_WORKERS: int = 4