import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .image_processing import ImageProcessor
from .inference import predict_both_models
from .model_loader import ModelManager
from .responses import (
    HealthResponse,
    PredictResponse,
    ErrorResponse,
    ModelPredictions,
    PredictionItem,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and processor
model_manager: ModelManager | None = None
image_processor: ImageProcessor | None = None

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global model_manager, image_processor
    
    # Startup
    logger.info("Starting up inference service...")
    
    # Load configuration from environment variables
    action_model_path = os.getenv("ACTION_MODEL_PATH", "/app/models/action_model.onnx")
    bodyparts_model_path = os.getenv("BODYPARTS_MODEL_PATH", "/app/models/bodyparts_model.onnx")
    image_size = int(os.getenv("IMAGE_SIZE", "224"))
    
    logger.info(f"Loading models from: action={action_model_path}, bodyparts={bodyparts_model_path}")
    logger.info(f"Using image size: {image_size}")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(
            action_model_path=action_model_path,
            bodyparts_model_path=bodyparts_model_path,
            image_size=image_size
        )
        
        # Pre-load models and labels
        for model_name in model_manager.get_all_models():
            model_manager.load_model(model_name)
            model_manager.load_labels(model_name)
        
        # Initialize image processor
        image_processor = ImageProcessor(target_size=image_size)
        
        logger.info("✓ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference service...")


# Create FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="Multi-label image classification API for action and bodyparts detection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for Cloud Run
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the service is running and models are loaded"
)
async def health_check():
    """Health check endpoint for Cloud Run."""
    try:
        models_loaded = model_manager is not None
        available_models = model_manager.get_all_models() if models_loaded else []
        
        return HealthResponse(
            status="healthy",
            models_loaded=models_loaded,
            available_models=available_models
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Image Classification",
    description="Upload an image to get action and bodyparts classifications",
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictResponse,
        },
        400: {
            "description": "Bad request (invalid image format)",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    }
)
async def predict(
    file: UploadFile = File(..., description="Image file to classify (JPEG, PNG, etc.)")
):
    """
    Predict action and bodyparts classifications for an uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Predictions from both action and bodyparts models
    """
    if not model_manager or not image_processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not initialized"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Expected image file."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.info(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")
        
        # Process image
        image_array = image_processor.process(image_bytes)
        
        # Run inference on both models
        results = predict_both_models(model_manager, image_array)
        
        # Format response
        response = _format_prediction_response(results, image_processor.target_size)
        
        logger.info(f"Prediction completed for {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


def _format_prediction_response(
    results: dict[str, dict[str, float]],
    image_size: int
) -> PredictResponse:
    """Format raw prediction results into structured response."""
    
    def _format_model_predictions(model_name: str, predictions: dict[str, float]) -> ModelPredictions:
        """Format predictions for a single model."""
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Create prediction items
        prediction_items = [
            PredictionItem(label=label, confidence=round(conf, 3))
            for label, conf in sorted_preds
        ]
        
        # Get top prediction
        top_prediction = prediction_items[0] if prediction_items else PredictionItem(
            label="unknown",
            confidence=0.0
        )
        
        return ModelPredictions(
            model_name=model_name,
            predictions=prediction_items,
            top_prediction=top_prediction
        )
    
    # Format predictions for both models
    action_predictions = _format_model_predictions("action", results.get("action", {}))
    bodyparts_predictions = _format_model_predictions("bodyparts", results.get("bodyparts", {}))
    
    return PredictResponse(
        success=True,
        action_model=action_predictions,
        bodyparts_model=bodyparts_predictions,
        image_size=image_size
    )


@app.get(
    "/",
    include_in_schema=False
)
async def root():
    """Root endpoint redirect to docs."""
    return JSONResponse({
        "message": "Image Classification API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    })


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", "8080")),
        log_level="info",
        access_log=True
    )
