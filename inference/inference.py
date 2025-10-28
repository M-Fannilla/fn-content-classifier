import logging
from typing import Dict
import numpy as np
from .model_loader import ModelManager, ModelsEnum

logger = logging.getLogger(__name__)

class Inference:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def run_inference(self, model: ModelsEnum, image_array: np.ndarray) -> dict[str, float]:
        """
        Run inference on a single model.

        Args:
            model: ModelEnum specifying which model to use
            image_array: Preprocessed image array

        Returns:
            Dictionary mapping labels to probabilities (rounded to 3 decimals)
        """

        # Load model and labels
        session = self.model_manager.models.get(model)
        labels = self.model_manager.labels.get(model)

        if not session or not labels:
            raise ValueError(f"Model or labels for '{model.value}' not loaded.")

        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_array})

        # Process outputs - apply sigmoid for multi-label classification
        image_probs = outputs[0].flatten()
        probabilities = 1 / (1 + np.exp(-image_probs))

        # Create label to probability mapping
        label_probs = {}
        for idx, prob in enumerate(probabilities):
            label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
            label_probs[label] = round(float(prob), 3)

        # Log top predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_labels = [labels[idx] if labels and idx < len(labels) else f"class_{idx}"
                      for idx in top_indices]

        logger.info(f"Top predictions for {model.value}: {top_labels}")

        return label_probs


    def predict_both_models(
            self,
            model_manager: ModelManager,
            image_array: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Run inference on both action and bodyparts models.

        Args:
            model_manager: ModelManager instance
            image_array: Preprocessed image array

        Returns:
            Dictionary with predictions from both models
        """
        results = {}

        for model_name in model_manager.get_all_models():
            try:
                results[model_name] = self.run_inference(model_name, image_array)
            except Exception as e:
                logger.error(f"Error running inference on {model_name}: {e}")
                results[model_name] = {}

        return results

