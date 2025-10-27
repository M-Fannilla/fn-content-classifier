from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from tqdm.auto import tqdm
import numpy as np
import wandb
import torch

# Import custom modules
from config import Config
from losses import WeightedBCELoss
from metrics import MultilabelMetrics
from training.model_helpers import create_model, setup_model_for_training
from utils import find_best_thresholds

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
            init_wandb: bool = True,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_columns = label_columns
        self.device = device
        self.class_freq = class_freq

        # Gradient accumulation settings
        self.grad_accum_steps = config.grad_accum_steps
        self.effective_batch_size = self.grad_accum_steps * self.config.batch_size

        # Learning Rate / Scheduler values
        self.learning_rate = self.config.learning_rate
        self.lr_range_start = self.config.lr_range_start
        self.lr_range_end = self.config.lr_range_end
        self.lr_range_steps = self.config.lr_range_steps

        # Loss / Optimizer / Scheduler
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)

        # AMP
        self.scaler = GradScaler(device)

        # Metrics
        self.metrics_calculator = MultilabelMetrics()
        self.reduce_metric = self.config.reduce_metric
        self.early_stop_metric = self.config.early_stop_metric
        self.best_model_metric = self.config.best_model_metric
        self.best_metric_value = 0.0
        self.best_model_state = None
        self.best_threshold = None

        # Early stopping
        self.early_stopping_counter = 0
        self.best_val_for_early_stopping = 0.0

        # Track thresholds per epoch
        self.validation_threshold = np.zeros(len(self.label_columns))

        # Performance history
        self.history = {
            k: []
            for k in [
                "train_loss",
                "val_loss",
                "val_f1_micro",
                "val_f1_macro",
                "val_roc_auc_micro",
                "val_roc_auc_macro",
            ]
        }
        self.init_wandb = init_wandb

        self.model = self.get_model().to(self.device)

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
                    'learning_rate': self.learning_rate,
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'img_size': self.config.img_size,
                    'early_stopping_patience': self.config.early_stopping_patience,
                    'early_stopping_min_delta': self.config.early_stopping_min_delta,
                    'lr_reduce_patience': self.config.lr_reduce_patience,
                    'lr_reduce_factor': self.config.lr_reduce_factor,
                    'lr_reduce_min_lr': self.config.lr_reduce_min_lr,
                    'weight_decay': self.config.weight_decay,
                    'bce_power': self.config.bce_power,
                    'tau_logit_adjust': self.config.tau_logit_adjust,
                    'num_classes': len(self.label_columns),
                    'effective_batch': self.effective_batch_size,
                    'micro_batch': self.config.batch_size,
                    'effective_batch_size': self.effective_batch_size,
                    'grad_accum_steps': self.grad_accum_steps,
                    'train_samples': len(self.train_loader.dataset),
                    'val_samples': len(self.val_loader.dataset),
                    'device': str(self.device)
                }
            )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True) # added
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')

        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Mixed precision forward pass
            with autocast(str(self.device)):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps

            # Mixed precision backward pass
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * self.grad_accum_steps
            total_loss += step_loss
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                loss=f"{step_loss:.4f}",
                avg=f"{total_loss/num_batches:.4f}"
            )

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
        self.model = self.get_model().to(self.device)

        if not self.learning_rate:
            self.learning_rate = self.find_lr()

        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer(learning_rate=self.learning_rate)
        self.scheduler = self.get_scheduler(optimizer=self.optimizer)

        # Initialize wandb if enabled
        if self.config.use_wandb:
            self._init_wandb()

        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(epoch=epoch)

            # Validation
            val_loss, val_metrics = self.validate_epoch()

            # Update learning rate based on validation F1 score
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
                    "lr_stage": "warmup" if epoch < self._warmup_epochs else "cosine",
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

    def find_lr(self) -> float:
        lr_finder = LRFinder(
            self.get_model(),
            self.get_optimizer(learning_rate=0.001), # dummy lr, will be overridden
            self.get_criterion(),
            device="cuda",
        )
        lr_finder.range_test(
            self.train_loader,
            start_lr=self.lr_range_start,
            end_lr=self.lr_range_end,
            num_iter=self.lr_range_steps,
            step_mode="exp",  # exponential increase
            smooth_f=0.05,  # light smoothing
            diverge_th=4,  # early stop if loss > 4x best
        )

        # Suggested LR:
        suggested_max_lr = (
            lr_finder.history["lr"][
                int(lr_finder.history["loss"].index(min(lr_finder.history["loss"])))
            ]
            * 0.5
        )
        print("Suggested Learning Rate: ", suggested_max_lr)

        return suggested_max_lr


    def info(self):
        print(f"Starting finetuning for {self.config.epochs} epochs...")
        print(f"  Model: {self.config.model_name}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  Early stopping patience: {self.config.early_stopping_patience}")
        print(f"  LR reduction patience: {self.config.lr_reduce_patience}")
        print(f"  Device: {self.device}")
        print(f"  Scheduler: ReduceLROnPlateau (patience={self.config.lr_reduce_patience})")
        print(f"  Early stopping: Enabled (patience={self.config.early_stopping_patience})")
        print("  Mixed precision: Enabled")
        print("-" * 50)

    def get_criterion(self)->WeightedBCELoss:
        return WeightedBCELoss(
            class_freq=self.class_freq,
            power=self.config.bce_power,
            device=self.device,
        ).to(self.device)

    def get_optimizer(self, learning_rate: float = None)->AdamW:
        return AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate if learning_rate else self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def get_model(self)->nn.Module:
        print(f"\nCreating model: {self.config.model_name}")
        model = create_model(self.config, num_classes=len(self.label_columns))
        return setup_model_for_training(
            self.config,
            model,
            device=self.device,
            class_freq=self.class_freq,
        )

    def get_scheduler(self, optimizer: AdamW)->SequentialLR:
        # Smart warmup (1 epoch for <=12 epochs, else 10%)
        main_epochs = self.config.epochs - self._warmup_epochs

        # Linear warmup from 1% → 100% LR
        warmup = LinearLR(
            optimizer,
            start_factor=self.config.linear_start_factor,  # 1% of base_lr at start
            end_factor=1.0,
            total_iters=self._warmup_epochs,
        )

        # Cosine learning rate decay (protect against overfitting)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=main_epochs,
            eta_min=self.learning_rate * self.config.cosine_annealing_min,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self._warmup_epochs]
        )

    @cached_property
    def _warmup_epochs(self) -> int:
        return max(1, min(3, self.config.epochs // 10))
