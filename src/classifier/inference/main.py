from src.classifier.inference.image_processing import ImageProcessor
from src.classifier.inference.inference import Inference
from src.classifier.inference.model_loader import ModelManager

model_manager = ModelManager()
model_manager.load_all()
image_processor = ImageProcessor()

inference = Inference(model_manager=model_manager)
