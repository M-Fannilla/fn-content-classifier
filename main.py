#!/usr/bin/env python3
"""
Main training script for multilabel image classification.

This script trains a ConvNeXt V2 model with specified configurations,
saves the model at best epochs and at the end of training.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Import our custom modules
from config import Config
from dataset import create_data_loaders, plot_label_distribution, load_and_prepare_data
from metrics import plot_roc_curves, plot_precision_recall_curves
from model import create_model, setup_model_for_training
from trainer import Trainer
from utils import visualize_predictions_from_dataloader, compute_class_frequency


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_system_info():
    """Print system information and configuration."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 60)


def create_output_directories(config: Config) -> None:
    """Create necessary output directories."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    print(f"Output directory created: {output_dir}")


def save_training_summary(
        config: Config,
        trainer: Trainer,
        history: dict,
        training_time: float
) -> None:
    """Save training summary and configuration."""
    summary = {
        'config': {
            'model_name': config.model_name,
            'model_type': config.model_type,
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'img_size': config.img_size,
            'threshold': config.threshold,
            'class_weight_method': config.class_weight_method,
            'loss_type': config.loss_type,
            'early_stopping_patience': config.early_stopping_patience,
            'lr_reduce_patience': config.lr_reduce_patience,
        },
        'training': {
            'total_epochs': len(history['train_loss']),
            'best_val_f1': trainer.best_val_f1,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'training_time_minutes': training_time / 60,
            'device': str(trainer.device),
        },
        'dataset': {
            'num_classes': len(trainer.label_columns),
            'class_names': trainer.label_columns,
            'train_samples': len(trainer.train_loader.dataset),
            'val_samples': len(trainer.val_loader.dataset),
        }
    }

    # Save as JSON
    import json
    summary_path = Path(config.output_dir) / "logs" / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to: {summary_path}")


def plot_and_save_results(
        config: Config,
        trainer: Trainer,
        history: dict,
        val_loader,
        original_labels: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray
) -> None:
    """Create and save various plots."""
    plots_dir = Path(config.output_dir) / "plots"

    # 1. Training history plot
    print("Creating training history plot...")
    trainer.plot_training_history(save_path=str(plots_dir / "training_history.png"))

    # 2. Label distribution plot
    print("Creating label distribution plot...")
    plot_label_distribution(
        original_labels=original_labels,
        train_labels=train_labels,
        test_labels=test_labels,
        label_columns=trainer.label_columns,
        save_path=str(plots_dir / "label_distribution.png")
    )

    # 3. ROC curves
    print("Creating ROC curves...")
    # Get validation predictions for ROC curves
    trainer.model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(trainer.device)
            outputs = trainer.model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()

            all_labels.append(labels.numpy())
            all_probabilities.append(probabilities)

    all_labels = np.concatenate(all_labels, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    plot_roc_curves(
        y_true=all_labels,
        y_pred_proba=all_probabilities,
        class_names=trainer.label_columns,
        save_path=str(plots_dir / "roc_curves.png")
    )

    # 4. Precision-Recall curves
    print("Creating Precision-Recall curves...")
    plot_precision_recall_curves(
        y_true=all_labels,
        y_pred_proba=all_probabilities,
        class_names=trainer.label_columns,
        save_path=str(plots_dir / "precision_recall_curves.png")
    )

    # 5. Sample predictions visualization
    print("Creating sample predictions visualization...")
    visualize_predictions_from_dataloader(
        model=trainer.model,
        dataloader=val_loader,
        label_columns=trainer.label_columns,
        threshold=config.threshold,
        num_samples=8
    )

    print(f"All plots saved to: {plots_dir}")


def main():
    """Main training function."""

    # Print system information
    print_system_info()
    config = Config()

    # Print configuration
    print("\nTRAINING CONFIGURATION")
    print("=" * 60)
    config.info()
    print("=" * 60)

    # Set random seed for reproducibility
    set_seed(config.seed)
    print(f"Random seed set to: {config.seed}")

    # Create output directories
    create_output_directories(config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create data loaders
    print("\nLoading dataset...")
    df, image_paths, labels = load_and_prepare_data(config=config)
    class_frequency = compute_class_frequency(df.drop(["file_name"], axis=1))
    (
        train_loader,
        val_loader,
        label_columns,
        original_labels,
        train_labels,
        test_labels
    ) = create_data_loaders(df, image_paths, labels, config)

    # Create model
    print(f"\nCreating model: {config.model_name}")
    model = create_model(config, num_classes=len(label_columns))
    model = setup_model_for_training(
        config, model, class_freq=class_frequency, device=device
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        class_freq=class_frequency,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        label_columns=label_columns,
        device=str(device)
    )

    # Print trainer information
    trainer.info()

    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    start_time = time.time()

    try:
        # Train model
        history = trainer.train(save_model=True)

        training_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total training time: {training_time / 60:.2f} minutes")
        print(f"Best validation {trainer.best_model_metric}: {trainer.best_metric_value:.4f}")
        print(f"Total epochs: {len(history['train_loss'])}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current model state...")
        print("Model saved successfully.")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()
