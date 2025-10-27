"""Model loading and management."""
import json
import logging
from typing import Dict, List
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ONNX models and their labels."""
    
    def __init__(
            self,
            action_model_path: str,
            bodyparts_model_path: str,
            image_size: int,
    ):
        self.models: Dict[str, ort.InferenceSession] = {}
        self.labels: Dict[str, List[str]] = {}
        self.image_size = image_size
        
        # Model configuration
        self.model_config = {
            "action": {
                "path": action_model_path,
                "labels_path": action_model_path.replace(".onnx", ".json"),
                "img_size": image_size
            },
            "bodyparts": {
                "path": bodyparts_model_path,
                "labels_path": bodyparts_model_path.replace(".onnx", ".json"),
                "img_size": image_size
            },
        }
    
    def load_labels(self, model_name: str) -> List[str]:
        """Load labels for a model from JSON file."""
        if model_name in self.labels:
            logger.info(f"Labels for '{model_name}' already loaded")
            return self.labels[model_name]
        
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        labels_path = self.model_config[model_name]["labels_path"]

        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            self.labels[model_name] = labels
            logger.info(f"Loaded {len(labels)} labels for '{model_name}'")
            return labels

        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return []
    
    def load_model(self, model_name: str) -> ort.InferenceSession:
        """Load an ONNX model into an inference session."""
        if model_name in self.models:
            logger.info(f"Model '{model_name}' already loaded")
            return self.models[model_name]
        
        if model_name not in self.model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_config[model_name]["path"]
        
        providers = ['CPUExecutionProvider']
        if ort.get_device().lower() == 'gpu':
            providers.insert(0, 'CUDAExecutionProvider')
        
        logger.info(f"Loading model '{model_name}' from {model_path}")
        
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            self.models[model_name] = session
            logger.info(f"Model '{model_name}' loaded successfully")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def get_all_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.model_config.keys())

