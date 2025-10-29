from .inference_config import InferenceConfig
from .image_processing import ImageProcessor
from .inference import Inference
from .model_loader import ModelManager, ModelsEnum

inference_config = InferenceConfig()

model_manager = ModelManager(ModelsEnum.ACTION)
model_manager.load_all()

inference = Inference(
    model_manager=model_manager,
    apply_threshold=True,
)

image_processor = ImageProcessor(
    target_size=model_manager.get_image_size(),
    batch_workers=inference_config.IMAGE_PROCESSING_WORKERS,
)


if __name__ == "__main__":
    image_to_predict = [
        '/Users/milosz/Projects/fn-content-dataset/compiled/1.jpg',
        # '/Users/milosz/Projects/3.webp',
        '/Users/milosz/Projects/21344319-onlyfans.jpg',
    ]
    np_images_batch = image_processor.process_batch(image_to_predict)

    results = inference.predict_all(
        images_paths=image_to_predict,
        image_array=np_images_batch,
    )

    # out = dict(zip(image_to_predict, results))

    from pprint import pprint
    pprint(results)