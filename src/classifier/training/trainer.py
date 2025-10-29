from pathlib import Path

from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import wandb
import torch

# Import custom modules
from .. import TORCH_MODELS_DIR, DEVICE
from ..model import ClassifierModel
from ..configs import TrainConfig, TorchModelConfig
from .losses import WeightedBCELoss
from .metrics import MultilabelMetrics
from .model_helpers import create_model, setup_model_for_training
from .utils import find_best_thresholds

class Trainer:
    """Training class for ConvNeXt V2 multilabel classification."""

    def __init__(
            self,
            config: TrainConfig,
            class_freq: np.ndarray,
            train_loader: DataLoader,
            val_loader: DataLoader,
            label_columns: list[str],
            init_wandb = True,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_columns = label_columns

        # Move model to device
        self.class_freq = class_freq
        self.model = self.get_model()

        self.criterion = WeightedBCELoss(
            class_freq=self.class_freq,
            power=self.config.bce_power,
        ).to(DEVICE)

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

        # Mixed precision scaler
        self.scaler = GradScaler(DEVICE)

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
            self.config.model_type,
            self.config.model_name,
            self.config.batch_size,
        ]
        return "_".join([str(c) for c in components])

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        # Initialize wandb
        if self.init_wandb and self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.best_model_name,
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
                    'train_samples': len(self.train_loader.dataset), # noqa
                    'val_samples': len(self.val_loader.dataset), # noqa
                    'device': str(DEVICE)
                }
            )

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE, non_blocking=True) # noqa
            labels = labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(DEVICE):
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

    def _validate_epoch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        y_pred, y_true, y_prob = [], [], []
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')

            for images, labels in pbar:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with autocast(DEVICE):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                y_pred.append(outputs.cpu().numpy())
                y_true.append(labels.cpu().numpy())
                y_prob.append(torch.sigmoid(outputs).cpu().numpy())

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

    def train(self, save_model: bool = True) -> None:
        """Main training loop with early stopping and learning rate reduction."""
        # Initialize wandb if enabled
        if self.config.use_wandb:
            self._init_wandb()

        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(epoch=epoch)

            # Validation
            val_loss, val_metrics = self.validate()
            self.scheduler.step()

            # Check for best model
            best_metric_value = val_metrics[self.best_model_metric]
            if best_metric_value > self.best_metric_value:
                self.best_metric_value = best_metric_value
                self.best_model_state = self.model.state_dict().copy()
                self.best_threshold = self.validation_threshold.copy()

                print(f"  → New best {self.best_model_metric}: {best_metric_value:.4f}")

                if save_model:
                    self.save_checkpoint(epoch=epoch)

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

        if save_model:
            self.save_checkpoint(epoch='final')
            self.save_onnx(epoch='final')

        # Finish wandb run
        if self.config.use_wandb:
            wandb.finish()

    def validate(self) -> tuple[float, dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()

        y_true, y_pred, y_prob, total_loss = self._validate_epoch()
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

    def _torch_model_config(self, epoch: int | str) -> tuple[Path, TorchModelConfig]:
        """Save the trained model."""
        epoch = str(epoch)
        filename = self.best_model_name
        model_path = (
            TORCH_MODELS_DIR / f"{epoch}_{filename}.pth" if epoch else f"{filename}.pth"
        )

        return model_path, TorchModelConfig(
            model_type=self.config.model_type,
            model_state_dict=self.best_model_state,
            labels=self.label_columns,
            image_size=self.config.img_size,
            threshold=self.best_threshold,
        )

    def save_onnx(self, epoch: int | str) -> TorchModelConfig:
        """Save the trained model."""
        model_path, model_config = self._torch_model_config(epoch=epoch)
        model_config.export_to_onnx(model=self.model)
        model_config.save_as_onnx_config()
        return model_config

    def save_checkpoint(self, epoch: int | str) -> TorchModelConfig:
        """Save the trained model."""
        model_path, model_config = self._torch_model_config(epoch)
        model_config.save_torch_model(save_path=str(model_path))
        return model_config

    def info(self) -> None:
        print(f"Starting finetuning for {self.config.epochs} epochs...")
        print(f"  Model: {self.config.model_name}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  Device: {DEVICE}")
        print("  Mixed precision: Enabled")
        print("-" * 50)

    def get_model(self) -> ClassifierModel:
        print(f"\nCreating model: {self.config.model_name}")
        model = create_model(self.config, num_classes=len(self.label_columns))
        model = setup_model_for_training(
            self.config, model, class_freq=self.class_freq
        )
        return model.to(DEVICE)