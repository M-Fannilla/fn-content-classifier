#!/usr/bin/env python3
"""
Main application with Gradio interface for FN Content Classifier.
Provides interactive image classification with probability visualization.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import warnings

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import InferenceEngine


class GradioInferenceApp:
    """Gradio application for interactive model inference."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        img_size: int = 224,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """Initialize the Gradio app with inference engine."""
        self.engine = InferenceEngine(
            model_path=model_path,
            config_path=config_path,
            img_size=img_size,
            threshold=threshold,
            device=device
        )
        
        # Get class names for display
        self.class_names = self.engine.get_class_names()
        
        print(f"Gradio app initialized with {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
    
    def classify_image(self, image: np.ndarray) -> Tuple[str, plt.Figure]:
        """
        Classify an image and return results with probability histogram.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (results_text, histogram_figure)
        """
        if image is None:
            return "Please upload an image.", None
        
        try:
            # Run inference
            probabilities, predictions = self.engine.predict_single(image)
            
            # Get predicted classes
            predicted_classes = [
                self.class_names[i] for i, pred in enumerate(predictions) 
                if pred == 1
            ]
            
            # Create results text
            results_text = f"**Predicted Classes:** {', '.join(predicted_classes) if predicted_classes else 'None'}\n\n"
            results_text += f"**Threshold:** {self.engine.threshold}\n\n"
            results_text += "**All Probabilities:**\n"
            
            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
                status = "✓" if predictions[i] == 1 else "✗"
                results_text += f"- {class_name}: {prob:.4f} {status}\n"
            
            # Create probability histogram
            fig = self.engine.create_probability_histogram(
                probabilities,
                title="Classification Probabilities"
            )
            
            return results_text, fig
            
        except Exception as e:
            error_msg = f"Error during classification: {str(e)}"
            print(error_msg)
            return error_msg, None
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(
            title="FN Content Classifier",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .main-header {
                text-align: center;
                margin-bottom: 20px;
            }
            """
        ) as interface:
            
            gr.HTML("""
                <div class="main-header">
                    <h1>🎯 FN Content Classifier</h1>
                    <p>Upload an image to classify adult content categories</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📸 Upload Image")
                    image_input = gr.Image(
                        label="Upload an image",
                        type="numpy",
                        height=300
                    )
                    
                    # Parameters section
                    gr.Markdown("### ⚙️ Parameters")
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=self.engine.threshold,
                        step=0.05,
                        label="Classification Threshold",
                        info="Higher values = more conservative predictions"
                    )
                    
                    classify_btn = gr.Button(
                        "🔍 Classify Image",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Results section
                    gr.Markdown("### 📊 Results")
                    results_text = gr.Markdown(
                        "Upload an image and click 'Classify Image' to see results.",
                        elem_id="results"
                    )
                    
                    # Histogram section
                    gr.Markdown("### 📈 Probability Distribution")
                    histogram_plot = gr.Plot(
                        label="Classification Probabilities",
                        show_label=True
                    )
            
            # Event handlers
            def update_threshold(new_threshold):
                """Update the threshold in the inference engine."""
                self.engine.threshold = new_threshold
                return f"Threshold updated to {new_threshold}"
            
            def classify_with_threshold(image, threshold):
                """Classify image with updated threshold."""
                if image is None:
                    return "Please upload an image.", None
                
                # Update threshold
                self.engine.threshold = threshold
                
                # Classify
                return self.classify_image(image)
            
            # Connect events
            threshold_slider.change(
                fn=update_threshold,
                inputs=[threshold_slider],
                outputs=[results_text]
            )
            
            classify_btn.click(
                fn=classify_with_threshold,
                inputs=[image_input, threshold_slider],
                outputs=[results_text, histogram_plot]
            )
            
            # Also classify on image upload
            image_input.change(
                fn=classify_with_threshold,
                inputs=[image_input, threshold_slider],
                outputs=[results_text, histogram_plot]
            )
            
            # Examples section
            gr.Markdown("### 💡 Tips")
            gr.Markdown("""
            - **Higher threshold (0.7-0.9)**: More conservative, fewer false positives
            - **Lower threshold (0.3-0.5)**: More sensitive, catches more cases
            - **Red bars**: Predicted classes (above threshold)
            - **Blue bars**: Non-predicted classes (below threshold)
            - The model can predict multiple classes simultaneously
            """)
        
        return interface


def main():
    """Main function to launch the Gradio app."""
    parser = argparse.ArgumentParser(description='FN Content Classifier - Gradio Interface')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model file (.pth)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to the config JSON file')
    
    # Optional arguments
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Default classification threshold (default: 0.5)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device to run inference on (auto-detect if not specified)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the server on (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the server on (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link for the interface')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        sys.exit(1)
    
    # Suppress warnings if not in debug mode
    if not args.debug:
        warnings.filterwarnings('ignore')
    
    try:
        # Create Gradio app
        print("Initializing Gradio app...")
        app = GradioInferenceApp(
            model_path=args.model_path,
            config_path=args.config_path,
            img_size=args.img_size,
            threshold=args.threshold,
            device=args.device
        )
        
        # Create interface
        interface = app.create_interface()
        
        # Launch interface
        print(f"Launching Gradio interface on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
