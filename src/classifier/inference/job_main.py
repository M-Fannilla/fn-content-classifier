import sys
import json
import logging
import time
from typing import List, Optional

from .configs import InferenceConfig
from .image_processing import ImageProcessor
from .inference import Inference
from .model_loader import ModelManager, ModelsEnum


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fn-content-classifier.job_main")


# ─────────────────────────────────────────────
# Model and inference setup
# ─────────────────────────────────────────────
try:
    start_init = time.time()
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


# ─────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────
def image_inference(image_to_predict: Optional[List[str]] = None):
    if not image_to_predict:
        logger.warning("No image URLs provided. Using default test image.")
        image_to_predict = [
            'https://www.porndos.com/images/thumb/7255.webp',
        ]

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


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()
    logger.info("🚀 Starting fn-content-classifier job")
    logger.info(f"CLI args: {sys.argv}")

    try:
        if len(sys.argv) < 2:
            raise ValueError("No URLs argument provided to job.")

        urls_json = sys.argv[1]
        urls = json.loads(urls_json)

        if not isinstance(urls, list):
            raise TypeError("URLs argument must be a JSON list of strings")

        logger.info(f"Received {len(urls)} URLs for inference")
        start = time.time()
        results = image_inference(image_to_predict=urls)
        elapsed = time.time() - start

        logger.info(f"✅ Inference completed successfully in {elapsed:.2f}s")
        logger.info(f"Results:\n{json.dumps(results, indent=2)}")

    except json.JSONDecodeError:
        logger.exception("❌ Failed to parse URLs JSON from command line")

    except Exception:
        logger.exception("❌ Unexpected error during job execution")

    end_time = time.time()
    logger.info("🏁 Job finished in %.2f seconds", end_time - start_time)