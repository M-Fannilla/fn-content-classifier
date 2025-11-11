"""Model loading and management."""
import logging

import numpy as np
import onnxruntime as ort

from .configs import OnnxModelConfig
from . import ONNX_DIR
from pathlib import Path

logger = logging.getLogger(__name__)

def find_files(directory: Path, extension: str) -> list[str]:
    """Find all files with a given extension in a directory."""
    return [p.name.removesuffix(extension) for p in directory.iterdir() if p.suffix == extension]

class ModelManager:
    """Manages ONNX models and their labels."""
    models: dict[str, ort.InferenceSession] = {}
    model_configs: dict[str, OnnxModelConfig] | None = None

    @staticmethod
    def _models_to_load() -> list[str]:
        return find_files(ONNX_DIR, '.onnx')

    @staticmethod
    def _configs_to_load() -> list[str]:
        return find_files(ONNX_DIR, '.json')

    def load_configs(self):
        configs = self._configs_to_load()

        if not configs:
            raise ValueError(f"No model configs found in {ONNX_DIR}")

        self.model_configs = {
            model_type:  OnnxModelConfig.load_config(f"{model_type}.json") for model_type in configs
        }

    def load_all(self):
        """Load all models and their labels."""
        self.load_configs()

        for model_name in self._models_to_load():
            self.load_model(model_name)

    def load_model(self, model: str) -> None:
        """Load an ONNX model into an inference session."""
        if model in self.models:
            return
        
        model_type = self.model_configs[model].model_type
        
        providers = ['CPUExecutionProvider']
        if ort.get_device().lower() == 'gpu':
            providers.insert(0, 'CUDAExecutionProvider')

        try:
            self.models[model] = ort.InferenceSession(ONNX_DIR / f'{model_type}.onnx', providers=providers)

        except Exception:
            logger.error(f"Model '{model_type}' from {ONNX_DIR} could not be loaded")

    def get_all_models(self) -> list[str]:
        """Get list of available model names."""
        return list(self.model_configs.keys())

    def get_image_size(self) -> int:
        """Get expected image size for a model."""
        image_size: int | None = None
        for config in self.model_configs.values():
            if not image_size:
                image_size = config.image_size
            else:
                if image_size != config.image_size:
                    raise ValueError("Inconsistent image sizes across models.")
        return image_size

    def get_threshold(self, model: str) -> np.ndarray:
        """Get labels for a model."""
        threshold = self.model_configs[model].threshold

        if not isinstance(threshold, np.ndarray):
            threshold = np.array(threshold, dtype=np.float32)

        return threshold

    def get_labels(self, model: str) -> list[str]:
        """Get labels for a model."""
        raw_labels = self.model_configs[model].labels
        return [l.replace(" ", "_") for l in raw_labels]

    def get_onnx_session(self, model: str) -> ort.InferenceSession:
        return self.models.get(model)
