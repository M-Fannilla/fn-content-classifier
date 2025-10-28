from dataclasses import dataclass, fields
from pathlib import Path

import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import wandb
import torch

# Import custom modules
from training.config import Config
from training.losses import WeightedBCELoss
from training.metrics import MultilabelMetrics
from training.model_helpers import create_model, setup_model_for_training
from training.utils import find_best_thresholds

@dataclass
class ModelConfig:
    model_name: str
    model_state_dict: dict
    labels: list[str]
    image_size: int
    tau_logit_adjust: float
    class_frequency: np.ndarray
    threshold: np.ndarray

    @classmethod
    def _config_kwargs(cls) -> list[str]:
        return [str(f.name) for f in fields(cls)]

    def save_config(self) -> dict:
        return {
            field: getattr(self, field) for field in self._config_kwargs()
        }

    @classmethod
    def load_model(cls, model_path: str) -> 'ModelConfig':
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model_dict = {
            field: checkpoint.get(field) for field in cls._config_kwargs()
        }
        return ModelConfig(**model_dict)


class Trainer:
    """Training class for ConvNeXt V2 multilabel classification."""

    def __init__(
            self,
            config: Config,
            class_freq: np.ndarray,
            train_loader: DataLoader,
            val_loader: DataLoader,
            label_columns: list[str],
            device: str,
            init_wandb = True,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_columns = label_columns
        self.device = device

        # Move model to device
        self.class_freq = class_freq
        self.model = self.get_model().to(self.device)

        self.criterion = WeightedBCELoss(
            class_freq=self.class_freq,
            power=self.config.bce_power,
            device=self.device,
        ).to(self.device)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.cosine_annealing_min * self.config.learning_rate,
        )

        self.warmup_epochs = max(1, int(0.1 * self.config.epochs))
        self.main_epochs = self.config.epochs - self.warmup_epochs

        # Mixed precision scaler
        self.scaler = GradScaler(device)

        # Metrics calculator
        self.metrics_calculator = MultilabelMetrics()

        # Best model tracking
        self.best_model_metric = config.best_model_metric
        self.best_metric_value = 0.0
        self.best_model_state = None

        self.validation_threshold = np.zeros(len(self.label_columns))
        self.best_threshold = np.zeros(len(self.label_columns))

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_micro': [],
            'val_f1_macro': [],
            'val_roc_auc_micro': [],
            'val_roc_auc_macro': []
        }

        self.init_wandb = init_wandb

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
        if self.init_wandb:
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
                    'weight_decay': self.config.weight_decay,
                    'bce_power': self.config.bce_power,
                    'tau_logit_adjust': self.config.tau_logit_adjust,
                    'num_classes': len(self.label_columns),
                    'train_samples': len(self.train_loader.dataset),
                    'val_samples': len(self.val_loader.dataset),
                    'device': str(self.device)
                }
            )

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

        y_true, y_pred, y_prob, total_loss = self.validate_single_epoch()
        self.validation_threshold = find_best_thresholds(
            y_true=y_true,
            y_prob=y_prob,
        )

        # Calculate metrics
        metrics = self.metrics_calculator.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_prob,
            threshold=self.validation_threshold,
        )

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, metrics

    def validate_single_epoch(
            self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        y_pred, y_true, y_prob = [], [], []
        total_loss, num_batches = 0.0, 0

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

                y_pred.append(predictions)
                y_true.append(labels_np)
                y_prob.append(probabilities)

                num_batches += 1

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / num_batches:.4f}",
                    }
                )

        # Concatenate all predictions
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)

        return y_true, y_pred, y_prob, total_loss

    def train(self, save_model: bool = True) -> dict[str, list[float]]:
        """Main training loop with early stopping and learning rate reduction."""
        # Initialize wandb if enabled
        if self.config.use_wandb:
            self._init_wandb()

        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(epoch=epoch)

            # Validation
            val_loss, val_metrics = self.validate_epoch()
            self.scheduler.step()

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
                    self.save_checkpoint_model(self.best_model_name, threshold=self.validation_threshold)

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

            # Clear GPU cache after each epoch to prevent memory accumulation
            torch.cuda.empty_cache()

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        print(f"\nLoaded best model with {self.best_model_metric}: {self.best_metric_value:.4f}")

        # Finish wandb run
        if self.config.use_wandb:
            wandb.finish()

        return self.history

    def save_checkpoint_model(self, filename: str, threshold: np.ndarray | float):
        """Save the trained model."""
        model_path = f"{self.config.models_dir}/checkpoint/{filename}.pth"
        self._save_pytorch(model_path=model_path, threshold=threshold)

    def save_best_model(self, filename: str, threshold: np.ndarray | float):
        """Save the trained model."""
        model_path = f"{self.config.models_dir}/{filename}.pth"
        self._save_pytorch(model_path=model_path, threshold=threshold)

    def _save_pytorch(self, model_path: str, threshold: np.ndarray | float):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        model_config = ModelConfig(
            model_name=self.config.model_name,
            model_state_dict=self.best_model_state,
            labels=self.label_columns,
            image_size=self.config.img_size,
            tau_logit_adjust=self.config.tau_logit_adjust,
            class_frequency=self.class_freq,
            threshold=threshold,
        )

        torch.save(
            model_config.save_config(),
            model_path
        )

        print(f"Saved PyTorch model to {model_path}")

    def info(self):
        print(f"Starting finetuning for {self.config.epochs} epochs...")
        print(f"  Model: {self.config.model_name}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  Device: {self.device}")
        print("  Mixed precision: Enabled")
        print("-" * 50)

    def get_model(self)->nn.Module:
        print(f"\nCreating model: {self.config.model_name}")
        model = create_model(self.config, num_classes=len(self.label_columns))
        model = setup_model_for_training(
            self.config, model, device=self.device, class_freq=self.class_freq
        )
        return model