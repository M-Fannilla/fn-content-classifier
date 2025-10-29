import argparse

from classifier.inference.inference_config import InferenceConfig
from .image_processing import ImageProcessor
from .inference import Inference
from .model_loader import ModelManager

inference_config = InferenceConfig()

model_manager = ModelManager()
model_manager.load_all()

inference = Inference(model_manager=model_manager)

image_processor = ImageProcessor(
    target_size=model_manager.get_image_size(),
    batch_workers=inference_config.IMAGE_PROCESSING_WORKERS,
)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--image_path",
        type=str,
        help="Image path to predict"
    )
    parsed_args = args.parse_args()
    image_to_predict = parsed_args.image_path

    np_images_batch = image_processor.process_batch([image_to_predict])
    results = inference.predict_all(np_images_batch)

    from pprint import pprint
    pprint(results)