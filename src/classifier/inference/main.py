import sys
import json
import logging
import time
from typing import List, Optional

from classifier.inference.file_loaders import GCPImageLoader
from .configs import InferenceConfig
from .image_processing import ImageProcessor
from .inference import Inference
from .model_loader import ModelManager, ModelsEnum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fn-content-classifier.job_main")

try:
    start_init = time.time()
    image_loader = GCPImageLoader()

    inference_config = InferenceConfig()
    model_manager = ModelManager(
        ModelsEnum.ACTION,
        ModelsEnum.BODYPARTS
    )
    model_manager.load_all()

    inference = Inference(
        model_manager=model_manager,
        apply_threshold=True,
    )

    image_processor = ImageProcessor(
        target_size=model_manager.get_image_size(),
        batch_workers=inference_config.IMAGE_PROCESSING_WORKERS,
    )
    logger.info(f"Initialization complete in {time.time() - start_init:.2f}s")

except Exception as e:
    logger.exception("❌ Failed during initialization")
    raise


def image_inference(image_to_predict: Optional[List[str]] = None):
    logger.info(f"Processing batch of {len(image_to_predict)} images")

    try:
        start_proc = time.time()
        np_images_batch = image_processor.process_batch(image_to_predict)
        logger.info(f"Image processing done in {time.time() - start_proc:.2f}s")

        start_infer = time.time()
        results = inference.predict_all(
            images_paths=image_to_predict,
            image_array=np_images_batch,
        )
        logger.info(f"Inference done in {time.time() - start_infer:.2f}s")
        return results

    except Exception:
        logger.exception("❌ Error during inference")
        raise


if __name__ == "__main__":
    logger.info("🚀 Starting fn-content-classifier job")
    logger.info(f"CLI args: {sys.argv}")

    try:
        if len(sys.argv) < 2:
            raise ValueError("No URLs argument provided to job.")

        urls_json = sys.argv[1]
        urls = json.loads(urls_json)

        if not isinstance(urls, list):
            raise TypeError("URLs argument must be a JSON list of strings")

        image_loader.download_images(urls)

        logger.info(f"Received {len(urls)} URLs for inference")
        results = image_inference(image_to_predict=urls)
        logger.info(f"Results:\n{json.dumps(results, indent=2)}")

    except json.JSONDecodeError:
        logger.exception("❌ Failed to parse URLs JSON from command line")

    except Exception:
        logger.exception("❌ Unexpected error during job execution")