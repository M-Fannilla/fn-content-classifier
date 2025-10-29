"""Model loading and management."""
import enum
import logging

import numpy as np
import onnxruntime as ort

from ..configs import OnnxModelConfig
from .. import ONNX_DIR

logger = logging.getLogger(__name__)


class ModelsEnum(enum.Enum):
    ACTION = "action"
    BODYPARTS = "bodyparts"



class ModelManager:
    """Manages ONNX models and their labels."""
    models: dict[ModelsEnum, ort.InferenceSession] = {}
    model_configs: dict[ModelsEnum, OnnxModelConfig] | None = None

    def __init__(self, *models_to_load: ModelsEnum):
        self.models_to_load = models_to_load

    def load_configs(self):
        self.model_configs = {
            model_type:  OnnxModelConfig.load_config(model_type.value) for model_type in self.models_to_load
        }

    def load_all(self):
        """Load all models and their labels."""
        self.load_configs()
        for model_name in self.models_to_load:
            self.load_model(model_name)

    def load_model(self, model: ModelsEnum) -> None:
        """Load an ONNX model into an inference session."""
        if model in self.models:
            return
        
        model_type = self.model_configs[model].model_type
        
        providers = ['CPUExecutionProvider']
        if ort.get_device().lower() == 'gpu':
            providers.insert(0, 'CUDAExecutionProvider')

        try:
            self.models[model] = ort.InferenceSession(ONNX_DIR / f"{model_type}.onnx", providers=providers)
            logger.info(f"Model '{model_type}' from {ONNX_DIR} loaded successfully")

        except Exception:
            logger.error(f"Model '{model_type}' from {ONNX_DIR} could not be loaded")

    def get_all_models(self) -> list[ModelsEnum]:
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

    def get_threshold(self, model: ModelsEnum) -> np.ndarray:
        """Get labels for a model."""
        threshold = self.model_configs[model].threshold

        if not isinstance(threshold, np.ndarray):
            threshold = np.array(threshold, dtype=np.float32)

        return threshold

    def get_labels(self, model: ModelsEnum) -> list[str]:
        """Get labels for a model."""
        return self.model_configs[model].labels

    def get_onnx_session(self, model: ModelsEnum) -> ort.InferenceSession:
        return self.models.get(model)
