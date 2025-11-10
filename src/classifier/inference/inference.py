from pathlib import Path

import logging
import numpy as np
from .model_loader import ModelManager

logger = logging.getLogger(__name__)

class Inference:
    def __init__(
            self,
            model_manager: ModelManager,
            apply_threshold: bool = True,
            threshold_offset: float = 0.0,
    ):
        self.model_manager = model_manager
        self.apply_threshold = apply_threshold
        self.threshold_offset = threshold_offset if apply_threshold else 0.0

    def predict(
            self,
            model: str,
            images_paths: list[Path],
            image_array: np.ndarray,
    ) -> dict[str, float]:
        session = self.model_manager.get_onnx_session(model)
        labels = self.model_manager.get_labels(model)
        thresholds = self.model_manager.get_threshold(model) + self.threshold_offset

        if not session or not labels:
            raise ValueError(f"Model or labels for '{model}' not loaded.")

        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_array})

        # Process outputs - apply sigmoid for multi-label classification
        image_probs = outputs[0]
        probabilities = 1 / (1 + np.exp(-image_probs))

        # Reduce probability to binary value of prediction per class
        probabilities = np.where(
            probabilities > thresholds, 1, 0
        )

        # Create label to probability mapping
        batch_probabilities = {}
        for ip, prob in zip(images_paths, probabilities):
            predicted_labels = []
            predictions = dict(zip(labels, prob.tolist()))
            for label, pred_value in predictions.items():
                if pred_value != 1:
                    continue
                predicted_labels.append(label)

            batch_probabilities[str(ip)] = predicted_labels

        return batch_probabilities


    def predict_all(self, images_paths: list[Path], image_array: np.ndarray) -> dict[str, float]:
        results = {}

        for model_name in self.model_manager.get_all_models():
            results[model_name] = self.predict(
                model=model_name,
                images_paths=images_paths,
                image_array=image_array,
            )

        return results

def merge_results(results: dict) -> dict:
    merged_results = {}

    for model_name, model_results in results.items():
        for image_path, labels in model_results.items():
            if image_path not in merged_results:
                merged_results[image_path] = set()
            merged_results[image_path].update(labels)

    # Convert sets back to lists
    for image_path in merged_results:
        merged_results[image_path] = list(merged_results[image_path])

    return merged_results

