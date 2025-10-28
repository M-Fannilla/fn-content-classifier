import random
import time
import numpy as np
import torch

# Import our custom modules
from training.config import Config
from training.dataset import create_data_loaders, load_and_prepare_data
from training.find_batch_size import find_batch_size, suggest_grad_accumulation
from training.trainer import Trainer
from training.utils import (
    compute_class_frequency,
    find_best_thresholds,
    print_system_info,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main training function."""

    # Print system information
    print_system_info()
    config = Config()

    # overwrite finetuning:
    config.num_epochs = 15
    config.bce_power = 0.6788091730324309
    config.tau_logit_adjust = 0.8612782621731778
    config.use_wandb = True

    # Print configuration
    print("\nTRAINING CONFIGURATION")
    print("=" * 60)
    config.info()
    print("=" * 60)

    # Set random seed for reproducibility
    set_seed(config.seed)
    print(f"Random seed set to: {config.seed}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create data loaders
    print("\nLoading dataset...")
    df, image_paths, labels = load_and_prepare_data(config=config)
    class_frequency = compute_class_frequency(df.drop(["file_name"], axis=1))

    if not config.batch_size:
        config.batch_size = find_batch_size(
            config=config,
            num_classes=len(labels),
            device=str(device),
        )
        config.grad_accum_steps = suggest_grad_accumulation(
            config.batch_size,
            target_eff_bs=1024,
        )

    (
        train_loader,
        val_loader,
        label_columns,
        original_labels,
        train_labels,
        test_labels
    ) = create_data_loaders(df, image_paths, labels, config)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
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
    history = trainer.train(save_model=True)
    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total training time: {training_time / 60:.2f} minutes")
    print(f"Best validation {trainer.best_model_metric}: {trainer.best_metric_value:.4f}")
    print(f"Total epochs: {len(history['train_loss'])}")

    y_true, y_pred, y_prob, total_loss = trainer.validate_single_epoch()
    dynamic_threshold = find_best_thresholds(y_true=y_true, y_prob=y_prob)

    trainer.save_best_model(
        filename=f'{trainer.best_model_name}',
        threshold=dynamic_threshold,
    )

if __name__ == "__main__":
    main()
