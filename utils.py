import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple


def visualize_predictions(
    model: torch.nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    label_columns: list[str],
    threshold: float = 0.5,
    figsize: tuple[int, int] = (16, 12),
    save_path: str = None
) -> None:
    """Visualize model predictions on test images.
    
    Args:
        model: Trained model
        test_images: Tensor of test images (N, C, H, W)
        test_labels: Tensor of true labels (N, num_classes)
        label_columns: List of class names
        threshold: Threshold for binary classification
        figsize: Figure size for the plot
        save_path: Optional path to save the plot
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move images to device
    test_images = test_images.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(test_images)
        probabilities = torch.sigmoid(predictions)
        predicted_labels = (probabilities > threshold).float()
    
    # Move to CPU for visualization
    test_images_cpu = test_images.cpu()
    true_labels_cpu = test_labels.cpu()
    predicted_labels_cpu = predicted_labels.cpu()
    probabilities_cpu = probabilities.cpu()
    
    num_samples = len(test_images_cpu)
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
    print("\nPrediction Summary:")
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
) -> None:
    """Visualize model predictions from a DataLoader.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test data
        label_columns: List of class names
        threshold: Threshold for binary classification
        num_samples: Number of random samples to visualize
    """
    # Collect all data from dataloader
    dataset = iter(dataloader.dataset)

    images, labels = [], []
    for s in range(num_samples):
        img, label = next(dataset)
        images.append(img)
        labels.append(label)

    test_images = torch.stack(images, dim=0)
    test_labels = torch.stack(labels, dim=0)

    visualize_predictions(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        label_columns=label_columns,
        threshold=threshold,
    )