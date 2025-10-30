from .configs import InferenceConfig
from .image_processing import ImageProcessor
from .inference import Inference
from .model_loader import ModelManager, ModelsEnum

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

def image_inference():
    image_to_predict = [
        'https://www.porndos.com/images/thumb/7255.webp',
    ]
    np_images_batch = image_processor.process_batch(image_to_predict)
    return inference.predict_all(
        images_paths=image_to_predict,
        image_array=np_images_batch,
    )

if __name__ == "__main__":
    from pprint import pprint

    results = image_inference()
    pprint(results)