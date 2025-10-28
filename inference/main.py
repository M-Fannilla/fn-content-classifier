from inference.image_processing import ImageProcessor
from inference.inference import Inference
from inference.model_loader import ModelManager

model_manager = ModelManager(
    action_model_name="action.onnx",
    bodyparts_model_name="bodyparts.onnx",
)
model_manager.load_all()
image_processor = ImageProcessor()

inference = Inference(model_manager=model_manager)
