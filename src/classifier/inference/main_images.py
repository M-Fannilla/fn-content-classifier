import json
import logging
from .download import GCPMediaDownloader
from .configs import InferenceConfig
from .image_processing import ImageProcessor
from .inference import ImageInference
from .model_loader import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fn-content-classifier.job_main")

media_loader = GCPMediaDownloader()
MODEL_MANAGER = ModelManager()
MODEL_MANAGER.load_all()

if __name__ == "__main__":
    urls = [
        'fn-ai-datasets/content-classification/v0/images/1000.jpg',
        'fn-ai-datasets/content-classification/v0/images/2001.jpg',
    ]

    image_inference = ImageInference(
        inference_config=InferenceConfig(),
        processor=ImageProcessor(target_size=MODEL_MANAGER.get_image_size()),
        model_manager=MODEL_MANAGER,
        apply_threshold=True,
        threshold_offset=0.0,
    )

    temp_files = media_loader.download_images(*urls)
    logger.info(f"Downloaded {len(temp_files)} images for inference")

    results = image_inference.inference(file_paths=temp_files)
    logger.info(f"Results:\n{json.dumps(results, indent=2)}")
