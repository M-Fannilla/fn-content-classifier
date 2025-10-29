"""Model loading and management."""
import enum
import logging

import onnxruntime as ort

from ..configs import OnnxModelConfig
from .. import ONNX_DIR

logger = logging.getLogger(__name__)


class ModelsEnum(enum.Enum):
    ACTION = "action"
    BODYPARTS = "bodyparts"



class ModelManager:
    """Manages ONNX models and their labels."""
    
    def __init__(self):
        self.models: dict[ModelsEnum, ort.InferenceSession] = {}
        self.model_configs = {
            ModelsEnum.ACTION: OnnxModelConfig.load_config('action'),
            ModelsEnum.BODYPARTS: OnnxModelConfig.load_config('bodyparts'),
        }

    def load_all(self):
        """Load all models and their labels."""
        for model_name in self.get_all_models():
            self.load_model(model_name)

    def load_model(self, model_enum: ModelsEnum) -> None:
        """Load an ONNX model into an inference session."""
        if model_enum in self.models:
            return
        
        model_type = self.model_configs[model_enum].model_type
        
        providers = ['CPUExecutionProvider']
        if ort.get_device().lower() == 'gpu':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.models[model_enum] = ort.InferenceSession(ONNX_DIR / f"{model_type}.onnx", providers=providers)
        logger.info(f"Model '{model_type}' from {ONNX_DIR} loaded successfully")

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

    def get_labels(self, model_enum: ModelsEnum) -> list[str]:
        """Get labels for a model."""
        return self.model_configs[model_enum].labels

