from pathlib import Path
import time
import logging
import numpy as np
from .configs import InferenceConfig
from .image_processing import ImageProcessor
from .model_loader import ModelManager
from .utils import flatten_labels, label_frequencies
from .video_processing import VideoProcessor

logger = logging.getLogger(__name__)


class ImageInference:
    def __init__(
            self,
            inference_config: InferenceConfig,
            processor: ImageProcessor | VideoProcessor,
            model_manager: ModelManager,
            apply_threshold: bool = True,
            threshold_offset: float = 0.0,
    ):
        self.inference_config = inference_config
        self.processor = processor
        self.model_manager = model_manager
        self.apply_threshold = apply_threshold
        self.threshold_offset = threshold_offset if apply_threshold else 0.0

    def inference(self, file_paths: list[Path]) -> dict[str, list[str]]:
        start_proc = time.time()
        images_batch = []
        for img_path in file_paths:
            img = self.processor.process(img_path)
            images_batch.append(img)

        np_images_batch = np.vstack(images_batch)
        logger.info(f"Image processing done in {time.time() - start_proc:.2f}s")

        results = self._model_inference(np_images_batch=np_images_batch)
        results = {str(url): res for url, res in zip(file_paths, results)}

        return results

    def _model_inference(self, np_images_batch: np.ndarray) -> list[list[str]]:
        results = []
        start_infer = time.time()
        for model_name in self.model_manager.get_all_models():
            results.append(
                self.predict(model=model_name, image_array=np_images_batch)
            )

        logger.info(f"Inference done in {time.time() - start_infer:.2f}s")
        return flatten_labels(results)

    def predict(
            self,
            model: str,
            image_array: np.ndarray,
            batch_size: int = 32,
    ) -> list[list[str]]:
        session = self.model_manager.get_onnx_session(model)
        labels = self.model_manager.get_labels(model)
        thresholds = self.model_manager.get_threshold(model) + self.threshold_offset

        if not session or not labels:
            raise ValueError(f"Model or labels for '{model}' not loaded.")

        # Run inference
        input_name = session.get_inputs()[0].name
        # Image array shape is (batch_size, channels, height, width)

        label_probabilities = []
        for start_idx in range(0, image_array.shape[0], batch_size):
            end_idx = start_idx + batch_size
            batch_array = image_array[start_idx:end_idx]

            outputs = session.run(None, {input_name: batch_array})

            # Process outputs - apply sigmoid for multi-label classification
            image_probs = outputs[0]
            probabilities = 1 / (1 + np.exp(-image_probs))

            # Reduce probability to binary value of prediction per class
            probabilities = np.where(probabilities > thresholds, 1, 0)

            label_probabilities.extend(probabilities)

        # Create label to probability mapping
        label_predictions = []
        for prob in label_probabilities:
            predicted_labels = []
            predictions = dict(zip(labels, prob.tolist()))
            for label, pred_value in predictions.items():
                if pred_value != 1:
                    continue
                predicted_labels.append(label)

            label_predictions.append(predicted_labels)

        return label_predictions


class VideoInference(ImageInference):

    def inference(self, file_path: Path) -> dict[str, list[str]]:
        start_proc = time.time()
        np_frames_batch = self.processor.process(file_path)
        logger.info(f"Video processing done in {time.time() - start_proc:.2f}s")

        results = self._model_inference(np_images_batch=np_frames_batch)
        freq = label_frequencies(results, min_count=1, min_percent=0.05)

        return {str(file_path): list(freq.keys())}
