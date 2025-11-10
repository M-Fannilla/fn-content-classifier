from pathlib import Path
import json
import logging
from .video_processing import VideoProcessor
from .configs import InferenceConfig
from .inference import VideoInference
from .model_loader import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fn-content-classifier.job_main")

MODEL_MANAGER = ModelManager()
MODEL_MANAGER.load_all()

if __name__ == "__main__":
    url = Path('/Users/milosz/Projects/fn-classifier-final/of.mp4')
    video_inference = VideoInference(
        inference_config=InferenceConfig(),
        processor=VideoProcessor(
            target_size=MODEL_MANAGER.get_image_size(),
            batch_workers=8,
            video_interval=24,
        ),
        model_manager=MODEL_MANAGER,
        apply_threshold=True,
        threshold_offset=0.0,
    )
    results = video_inference.inference(file_path=url)
    logger.info(f"Results:\n{json.dumps(results, indent=2)}")
