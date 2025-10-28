from pydantic import BaseModel, Field



class PredictionItem(BaseModel):
    """Single prediction item with label and confidence."""
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class ModelPredictions(BaseModel):
    """Predictions from a single model."""
    model_name: str = Field(..., description="Name of the model (action or bodyparts)")
    predictions: list[PredictionItem] = Field(..., description="List of predictions sorted by confidence")
    top_prediction: PredictionItem = Field(..., description="Highest confidence prediction")


class PredictResponse(BaseModel):
    """Complete prediction response from both models."""
    success: bool = Field(..., description="Whether the prediction was successful")
    action_model: ModelPredictions = Field(..., description="Predictions from action classification model")
    bodyparts_model: ModelPredictions = Field(..., description="Predictions from bodyparts classification model")
    image_size: int = Field(..., description="Input image size used for inference")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    available_models: list[str] = Field(..., description="List of available model names")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")