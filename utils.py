import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import torchvision.transforms as T
from PIL import Image


def visualize_predictions(
    model: torch.nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    label_columns: List[str],
    threshold: float = 0.5,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> None:
    """Visualize model predictions on test images.
    
    Args:
        model: Trained model
        test_images: Tensor of test images (N, C, H, W)
        test_labels: Tensor of true labels (N, num_classes)
        label_columns: List of class names
        threshold: Threshold for binary classification
        num_samples: Number of random samples to visualize
        figsize: Figure size for the plot
        save_path: Optional path to save the plot
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move images to device
    test_images = test_images.to(device)
    
    # Get random indices
    total_samples = len(test_images)
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"Warning: Requested {num_samples} samples but only {total_samples} available. Showing all.")
    
    random_indices = random.sample(range(total_samples), num_samples)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(test_images[random_indices])
        probabilities = torch.sigmoid(predictions)
        predicted_labels = (probabilities > threshold).float()
    
    # Move to CPU for visualization
    test_images_cpu = test_images[random_indices].cpu()
    true_labels_cpu = test_labels[random_indices].cpu()
    predicted_labels_cpu = predicted_labels.cpu()
    probabilities_cpu = probabilities.cpu()
    
    # Create subplots
    rows = (num_samples + 3) // 4  # 4 images per row
    cols = min(4, num_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i in range(num_samples):
        ax = axes_flat[i]
        
        # Display image
        image = test_images_cpu[i]
        # Denormalize image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_display = image * std + mean
        image_display = torch.clamp(image_display, 0, 1)
        
        ax.imshow(image_display.permute(1, 2, 0))
        ax.axis('off')
        
        # Get true and predicted labels
        true_labels = true_labels_cpu[i]
        pred_labels = predicted_labels_cpu[i]
        probs = probabilities_cpu[i]
        
        # Create title with true and predicted labels
        true_classes = [label_columns[j] for j in range(len(label_columns)) if true_labels[j] == 1]
        pred_classes = [label_columns[j] for j in range(len(label_columns)) if pred_labels[j] == 1]
        
        # Create detailed prediction text
        prediction_text = f"Sample {i+1}\n"
        prediction_text += f"True: {', '.join(true_classes) if true_classes else 'None'}\n"
        prediction_text += f"Pred: {', '.join(pred_classes) if pred_classes else 'None'}\n"
        
        # Add confidence scores for predicted classes
        if pred_classes:
            prediction_text += "Confidence:\n"
            for j, class_name in enumerate(pred_classes):
                class_idx = label_columns.index(class_name)
                conf = probs[class_idx].item()
                prediction_text += f"  {class_name}: {conf:.3f}\n"
        
        ax.set_title(prediction_text, fontsize=8, pad=10)
        
        # Color code the border based on prediction accuracy
        correct = torch.equal(true_labels, pred_labels)
        border_color = 'green' if correct else 'red'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.suptitle(f'Model Predictions (Threshold: {threshold})\n'
                f'Green border = Correct, Red border = Incorrect', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    correct_predictions = 0
    for i in range(num_samples):
        true_labels = true_labels_cpu[i]
        pred_labels = predicted_labels_cpu[i]
        if torch.equal(true_labels, pred_labels):
            correct_predictions += 1
    
    accuracy = correct_predictions / num_samples
    print(f"\nPrediction Summary:")
    print(f"  Samples shown: {num_samples}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Threshold used: {threshold}")


def visualize_predictions_from_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    label_columns: List[str],
    threshold: float = 0.5,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> None:
    """Visualize model predictions from a DataLoader.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test data
        label_columns: List of class names
        threshold: Threshold for binary classification
        num_samples: Number of random samples to visualize
        figsize: Figure size for the plot
        save_path: Optional path to save the plot
    """
    # Collect all data from dataloader
    all_images = []
    all_labels = []
    
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
    
    # Concatenate all batches
    test_images = torch.cat(all_images, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    
    # Use the main visualization function
    visualize_predictions(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        label_columns=label_columns,
        threshold=threshold,
        num_samples=num_samples,
        figsize=figsize,
        save_path=save_path
    )


def get_prediction_confidence(
    model: torch.nn.Module,
    test_images: torch.Tensor,
    label_columns: List[str],
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get prediction confidence scores for test images.
    
    Args:
        model: Trained model
        test_images: Tensor of test images (N, C, H, W)
        label_columns: List of class names
        threshold: Threshold for binary classification
        
    Returns:
        Tuple of (predictions, probabilities, confidence_scores)
    """
    model.eval()
    device = next(model.parameters()).device
    
    test_images = test_images.to(device)
    
    with torch.no_grad():
        predictions = model(test_images)
        probabilities = torch.sigmoid(predictions)
        predicted_labels = (probabilities > threshold).float()
    
    # Calculate confidence as max probability for each sample
    confidence_scores = probabilities.max(dim=1)[0].cpu().numpy()
    
    return (
        predicted_labels.cpu().numpy(),
        probabilities.cpu().numpy(),
        confidence_scores
    )


def analyze_prediction_errors(
    model: torch.nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    label_columns: List[str],
    threshold: float = 0.5,
    top_k: int = 10
) -> None:
    """Analyze prediction errors and show most confident wrong predictions.
    
    Args:
        model: Trained model
        test_images: Tensor of test images (N, C, H, W)
        test_labels: Tensor of true labels (N, num_classes)
        label_columns: List of class names
        threshold: Threshold for binary classification
        top_k: Number of top errors to show
    """
    model.eval()
    device = next(model.parameters()).device
    
    test_images = test_images.to(device)
    
    with torch.no_grad():
        predictions = model(test_images)
        probabilities = torch.sigmoid(predictions)
        predicted_labels = (probabilities > threshold).float()
    
    # Move to CPU
    test_images_cpu = test_images.cpu()
    true_labels_cpu = test_labels.cpu()
    predicted_labels_cpu = predicted_labels.cpu()
    probabilities_cpu = probabilities.cpu()
    
    # Find incorrect predictions
    correct_mask = torch.equal(true_labels_cpu, predicted_labels_cpu, dim=1)
    incorrect_indices = torch.where(~correct_mask)[0]
    
    if len(incorrect_indices) == 0:
        print("No incorrect predictions found!")
        return
    
    # Get confidence scores for incorrect predictions
    incorrect_confidences = probabilities_cpu[incorrect_indices].max(dim=1)[0]
    
    # Get top-k most confident wrong predictions
    top_k = min(top_k, len(incorrect_indices))
    top_errors = torch.topk(incorrect_confidences, top_k)
    top_error_indices = incorrect_indices[top_errors.indices]
    
    print(f"Found {len(incorrect_indices)} incorrect predictions out of {len(test_images)} total.")
    print(f"Showing top {top_k} most confident wrong predictions:\n")
    
    for i, idx in enumerate(top_error_indices):
        true_labels = true_labels_cpu[idx]
        pred_labels = predicted_labels_cpu[idx]
        probs = probabilities_cpu[idx]
        
        true_classes = [label_columns[j] for j in range(len(label_columns)) if true_labels[j] == 1]
        pred_classes = [label_columns[j] for j in range(len(label_columns)) if pred_labels[j] == 1]
        
        print(f"Error {i+1} (Sample {idx.item()}):")
        print(f"  True: {', '.join(true_classes) if true_classes else 'None'}")
        print(f"  Pred: {', '.join(pred_classes) if pred_classes else 'None'}")
        print(f"  Confidence: {probs.max().item():.3f}")
        print()
