import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Optional
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import ConvNeXtV2MultilabelClassifier


class InferenceEngine:
    """Inference engine for the FN Content Classifier model."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        img_size: int = 224,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the saved model file
            config_path: Path to the config JSON file
            img_size: Input image size for the model
            threshold: Classification threshold for binary predictions
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.img_size = img_size
        self.threshold = threshold
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Set up image preprocessing
        self.transform = self._get_transform()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _load_model(self, model_path: str) -> ConvNeXtV2MultilabelClassifier:
        """Load the trained model."""
        # Create model instance
        model = ConvNeXtV2MultilabelClassifier(
            model_name=self.config['model_name'],
            num_classes=len(self.config['label_columns']),
            pretrained=False  # We're loading our own weights
        )
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        print(f"Model loaded from: {model_path}")
        print(f"Number of classes: {len(self.config['label_columns'])}")
        
        return model
    
    def _get_transform(self):
        """Get image preprocessing transform."""
        import torchvision.transforms as transforms
        
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for model inference."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        return image_tensor.unsqueeze(0)
    
    def predict_single(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for a single image.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            binary_predictions = (probabilities > self.threshold).astype(int)
        
        return probabilities, binary_predictions
    
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        # Preprocess all images
        image_tensors = []
        for image in images:
            image_tensor = self.preprocess_image(image)
            image_tensors.append(image_tensor)
        
        # Stack into batch
        batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            binary_predictions = (probabilities > self.threshold).astype(int)
        
        return probabilities, binary_predictions
    
    def predict_video(self, video_path: str, frame_skip: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Predict labels for video frames.
        
        Args:
            video_path: Path to video file
            frame_skip: Number of frames to skip between predictions
            
        Returns:
            List of (probabilities, binary_predictions) for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {frame_count} frames, {fps:.2f} FPS")
        
        predictions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % frame_skip == 0:
                try:
                    probs, preds = self.predict_single(frame)
                    predictions.append((probs, preds))
                    print(f"Processed frame {frame_idx}/{frame_count}")
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    continue
            
            frame_idx += 1
        
        cap.release()
        return predictions
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.config['label_columns']
    
    def create_probability_histogram(
        self,
        probabilities: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Classification Probabilities"
    ) -> plt.Figure:
        """Create a histogram of classification probabilities."""
        class_names = self.get_class_names()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar(range(len(class_names)), probabilities)
        
        # Color bars based on threshold
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > self.threshold:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        # Customize plot
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.axhline(y=self.threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({self.threshold})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add probability values on bars
        for i, prob in enumerate(probabilities):
            ax.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogram saved to: {save_path}")
        
        return fig


def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(description='FN Content Classifier Inference')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model file (.pth)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to the config JSON file')
    
    # Input arguments
    parser.add_argument('--image_paths', type=str, nargs='+',
                       help='Paths to input images')
    parser.add_argument('--video_path', type=str,
                       help='Path to input video file')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device to run inference on (auto-detect if not specified)')
    
    # Output arguments
    parser.add_argument('--output_probs', action='store_true',
                       help='Output probability values')
    parser.add_argument('--save_histogram', type=str,
                       help='Save probability histogram to specified path')
    parser.add_argument('--frame_skip', type=int, default=30,
                       help='Frames to skip between video predictions (default: 30)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_paths and not args.video_path:
        parser.error("Either --image_paths or --video_path must be specified")
    
    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")
    
    if not os.path.exists(args.config_path):
        parser.error(f"Config file not found: {args.config_path}")
    
    # Initialize inference engine
    print("Initializing inference engine...")
    engine = InferenceEngine(
        model_path=args.model_path,
        config_path=args.config_path,
        img_size=args.img_size,
        threshold=args.threshold,
        device=args.device
    )
    
    # Process images
    if args.image_paths:
        print(f"\nProcessing {len(args.image_paths)} images...")
        
        for i, image_path in enumerate(args.image_paths):
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            print(f"\n--- Image {i+1}: {os.path.basename(image_path)} ---")
            
            # Load and predict
            image = Image.open(image_path)
            probabilities, predictions = engine.predict_single(image)
            
            # Get class names
            class_names = engine.get_class_names()
            
            # Print results
            predicted_classes = [class_names[j] for j, pred in enumerate(predictions) if pred == 1]
            print(f"Predicted classes: {predicted_classes}")
            
            if args.output_probs:
                print("Probabilities:")
                for j, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                    status = "✓" if predictions[j] == 1 else "✗"
                    print(f"  {class_name}: {prob:.4f} {status}")
            
            # Save histogram for first image if requested
            if args.save_histogram and i == 0:
                engine.create_probability_histogram(
                    probabilities,
                    save_path=args.save_histogram,
                    title=f"Classification Probabilities - {os.path.basename(image_path)}"
                )
    
    # Process video
    if args.video_path:
        if not os.path.exists(args.video_path):
            print(f"Warning: Video not found: {args.video_path}")
        else:
            print(f"\nProcessing video: {os.path.basename(args.video_path)}")
            
            try:
                predictions = engine.predict_video(args.video_path, frame_skip=args.frame_skip)
                print(f"Processed {len(predictions)} frames")
                
                if args.output_probs and predictions:
                    # Show results for first frame
                    probabilities, binary_preds = predictions[0]
                    class_names = engine.get_class_names()
                    predicted_classes = [class_names[j] for j, pred in enumerate(binary_preds) if pred == 1]
                    print(f"First frame predicted classes: {predicted_classes}")
                    
                    if args.output_probs:
                        print("First frame probabilities:")
                        for j, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                            status = "✓" if binary_preds[j] == 1 else "✗"
                            print(f"  {class_name}: {prob:.4f} {status}")
                
            except Exception as e:
                print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
