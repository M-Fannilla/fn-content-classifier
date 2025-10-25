import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb
from config import Config
from losses import WeightedBCELoss
from metrics import MultilabelMetrics
from utils import find_best_thresholds


class Trainer:
    """Training class for ConvNeXt V2 multilabel classification."""

    def __init__(
            self,
            model: nn.Module,
            config: Config,
            class_freq: np.ndarray,
            train_loader: DataLoader,
            val_loader: DataLoader,
            label_columns: list[str],
            device: str
    ):

        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_columns = label_columns
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Setup loss function
        self.criterion = WeightedBCELoss(
            class_freq=class_freq,
            power=config.bce_power,
            device=device,
        ).to(device)

        # Setup optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler - use ReduceLROnPlateau for better control
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.lr_reduce_factor,
            patience=config.lr_reduce_patience,
            min_lr=config.lr_reduce_min_lr,
        )

        # Mixed precision scaler
        self.scaler = GradScaler(device)

        # Metrics calculator
        self.metrics_calculator = MultilabelMetrics(
            threshold=config.threshold
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_micro': [],
            'val_f1_macro': [],
            'val_roc_auc_micro': [],
            'val_roc_auc_macro': []
        }

        self.reduce_metric = config.reduce_metric
        self.early_stop_metric = config.early_stop_metric

        # Best model tracking
        self.best_model_metric = config.best_model_metric
        self.best_metric_value = 0.0
        self.best_model_state = None

        # Early stopping tracking
        self.early_stopping_counter = 0
        self.best_val_for_early_stopping = 0.0

        self.validation_threshold = np.zeros(len(self.label_columns))
        self.best_threshold = np.zeros(len(self.label_columns))

        # Initialize wandb if enabled
        if config.use_wandb:
            self._init_wandb()

    @property
    def best_model_name(self):
        components = [
            "wbce",
            self.config.model_type,
            self.config.model_name,
            self.config.batch_size,
            self.config.img_size,
        ]
        return "_".join([str(c) for c in components])

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        # Generate run name if not provided
        run_name = self.best_model_name

        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            tags=self.config.wandb_tags or [],
            config={
                'model_name': self.config.model_name,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'img_size': self.config.img_size,
                'early_stopping_patience': self.config.early_stopping_patience,
                'early_stopping_min_delta': self.config.early_stopping_min_delta,
                'lr_reduce_patience': self.config.lr_reduce_patience,
                'lr_reduce_factor': self.config.lr_reduce_factor,
                'lr_reduce_min_lr': self.config.lr_reduce_min_lr,
                'class_weight_method': self.config.class_weight_method,
                'loss_type': self.config.loss_type,
                'num_classes': len(self.label_columns),
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'device': str(self.device)
            }
        )

        # Log model architecture
        # wandb.watch(self.model, log="all", log_freq=100)

    def _log_metrics(self, metrics: dict[str, float], epoch: int, prefix: str = ""):
        """Log metrics to wandb."""
        if self.config.use_wandb:
            log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
            log_dict['epoch'] = epoch
            wandb.log(log_dict)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(str(self.device)):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })

        return total_loss / num_batches

    def validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')

            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(str(self.device)):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                # Store predictions and labels
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_predictions.append(predictions)
                all_labels.append(labels_np)
                all_probabilities.append(probabilities)

                num_batches += 1
                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / num_batches:.4f}",
                    }
                )

        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)

        # Calculate metrics
        metrics = self.metrics_calculator.compute_metrics(
            all_labels, all_predictions, all_probabilities
        )

        self.validation_threshold = find_best_thresholds(
            all_predictions,
            all_probabilities,
        )

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, metrics

    def train(self, save_model: bool = True) -> dict[str, list[float]]:
        """Main training loop with early stopping and learning rate reduction."""
        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Training
            train_loss = self.train_epoch(epoch=epoch)

            # Validation
            val_loss, val_metrics = self.validate_epoch()

            # Update learning rate based on validation F1 score
            self.scheduler.step(val_metrics[self.reduce_metric])

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_micro'].append(val_metrics['f1_micro'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])
            self.history['val_roc_auc_micro'].append(val_metrics.get('roc_auc_micro', 0.0))
            self.history['val_roc_auc_macro'].append(val_metrics.get('roc_auc_macro', 0.0))

            # Check for best model
            best_metric_value = val_metrics[self.best_model_metric]
            if best_metric_value > self.best_metric_value:
                self.best_metric_value = best_metric_value
                self.best_model_state = self.model.state_dict().copy()
                self.best_threshold = self.validation_threshold.copy()

                print(f"  → New best {self.best_model_metric}: {best_metric_value:.4f}")
                if save_model:
                    self.save_model(self.best_model_name)

            # Log metrics to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1_micro': val_metrics['f1_micro'],
                    'val_f1_macro': val_metrics['f1_macro'],
                    'val_f1_samples': val_metrics['f1_samples'],
                    'val_precision_micro': val_metrics['precision_micro'],
                    'val_precision_macro': val_metrics['precision_macro'],
                    'val_recall_micro': val_metrics['recall_micro'],
                    'val_recall_macro': val_metrics['recall_macro'],
                    'val_roc_auc_micro': val_metrics.get('roc_auc_micro', 0.0),
                    'val_roc_auc_macro': val_metrics.get('roc_auc_macro', 0.0),
                    'val_pr_auc_micro': val_metrics.get('pr_auc_micro', 0.0),
                    'val_pr_auc_macro': val_metrics.get('pr_auc_macro', 0.0),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    f'best_{self.best_model_metric}': self.best_metric_value
                })

            # Early stopping check
            early_stopping_metric = val_metrics[self.early_stop_metric]
            if early_stopping_metric > self.best_val_for_early_stopping + self.config.early_stopping_min_delta:
                self.best_val_for_early_stopping = early_stopping_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Clear GPU cache after each epoch to prevent memory accumulation
            torch.cuda.empty_cache()

            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                print(f"Best {self.best_model_metric} Micro: {self.best_metric_value:.4f}")
                print(f"No improvement for {self.config.early_stopping_patience} epochs.")
                break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nLoaded best model with {self.best_model_metric}: {self.best_metric_value:.4f}")

        # Finish wandb run
        if self.config.use_wandb:
            wandb.finish()

        return self.history

    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1 Micro plot
        axes[0, 1].plot(self.history['val_f1_micro'], label='F1 Micro', color='green')
        axes[0, 1].set_title('Validation F1 Micro')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Macro plot
        axes[1, 0].plot(self.history['val_f1_macro'], label='F1 Macro', color='orange')
        axes[1, 0].set_title('Validation F1 Macro')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # ROC AUC plot
        axes[1, 1].plot(self.history['val_roc_auc_micro'], label='ROC AUC Micro', color='red')
        axes[1, 1].plot(self.history['val_roc_auc_macro'], label='ROC AUC Macro', color='purple')
        axes[1, 1].set_title('Validation ROC AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('ROC AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_model(self, path: str):
        """Save the trained model."""
        # Create directory if it doesn't exist
        model_path = f"./{self.config.output_dir}/{path}"

        if os.path.dirname(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self._save_pytorch(filename=model_path + ".pth")

        print(f"Model saved to {path}")

    def _save_pytorch(self, filename: str = "convnextv2_finetuned.pth"):
        torch.save({
            'model_name': self.config.model_name,
            'model_state_dict': self.best_model_state,
            'labels_columns': self.label_columns,
            'image_size': self.config.img_size,
            'threshold': self.best_threshold,  # Keep static threshold for backward compatibility
        }, filename)

        print(f"Saved PyTorch model to {filename}")

    def info(self):
        print(f"Starting finetuning for {self.config.epochs} epochs...")
        print(f"  Model: {self.config.model_name}")
        print(f"  Learning rate: {self.config.learning_rate}")

        print(f"  Loss function: {self.config.loss_type}")
        print(f"  Class weight method: {self.config.class_weight_method}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")

        print(f"  Early stopping patience: {self.config.early_stopping_patience}")
        print(f"  LR reduction patience: {self.config.lr_reduce_patience}")
        print(f"  Device: {self.device}")

        print(f"  Scheduler: ReduceLROnPlateau (patience={self.config.lr_reduce_patience})")
        print(f"  Early stopping: Enabled (patience={self.config.early_stopping_patience})")
        print("  Mixed precision: Enabled")
        print("-" * 50)
