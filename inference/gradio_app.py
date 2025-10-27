"""
Gradio application for ONNX model inference.
Upload images and get predictions from action and bodyparts models.
"""
import os
import logging
from pathlib import Path
import gradio as gr
from model_loader import ModelManager
from image_processing import preprocess_image
from inference import predict_both_models
from visualization import create_combined_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model manager
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
model_manager = ModelManager(MODEL_DIR)

# Preload models at startup
logger.info("Preloading models...")
try:
    model_manager.load_model("action")
    model_manager.load_model("bodyparts")
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {e}")


def predict_image(image, top_k):
    """Process uploaded image and return predictions."""
    if image is None:
        return None, "Please upload an image"
    
    try:
        # Preprocess image
        logger.info("Preprocessing image...")
        image_array = preprocess_image(image, target_size=384)
        
        # Run inference on both models
        logger.info("Running inference...")
        results = predict_both_models(model_manager, image_array)
        
        # Create visualization
        logger.info("Creating visualization...")
        fig = create_combined_plot(
            results.get("action", {}),
            results.get("bodyparts", {}),
            top_k=top_k
        )
        
        # Format results as text
        results_text = "## Predictions\n\n"
        
        if "action" in results:
            results_text += "### Action Model\n"
            sorted_action = sorted(results["action"].items(), key=lambda x: x[1], reverse=True)[:5]
            for label, prob in sorted_action:
                results_text += f"- **{label}**: {prob:.3f}\n"
            results_text += "\n"
        
        if "bodyparts" in results:
            results_text += "### Bodyparts Model\n"
            sorted_bodyparts = sorted(results["bodyparts"].items(), key=lambda x: x[1], reverse=True)[:5]
            for label, prob in sorted_bodyparts:
                results_text += f"- **{label}**: {prob:.3f}\n"
        
        return fig, results_text
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="ONNX Model Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎯 Image Classifier
        
        Upload a **JPG** or **PNG** image to get predictions from both **Action** and **Bodyparts** models.
        
        The models will analyze the image and return probability scores for various classifications.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            image_input = gr.Image(
                type="pil",
                label="Upload Image (JPG/PNG)",
                height=400
            )
            
            top_k_slider = gr.Slider(
                minimum=5,
                maximum=15,
                value=10,
                step=1,
                label="Number of top predictions to show",
            )
            
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### 📝 Instructions
                1. Upload or paste a JPG/PNG image
                2. Adjust the number of predictions to display
                3. Click "Analyze Image"
                4. View results in the chart and text summary
                """
            )
        
        with gr.Column(scale=2):
            # Outputs
            plot_output = gr.Plot(label="Prediction Results")
            text_output = gr.Markdown(label="Top Predictions")
    
    # Examples section (removed for now due to URL input compatibility)
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict_image,
        inputs=[image_input, top_k_slider],
        outputs=[plot_output, text_output]
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )

