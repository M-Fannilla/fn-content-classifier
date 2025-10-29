import argparse
import time
import torch

# Import our custom modules
from .configs import TrainConfig
from .training.dataset import create_data_loaders, load_and_prepare_data
from .training.trainer import Trainer
from .training.utils import (
    compute_class_frequency,
    print_system_info,
    set_seed,
)

def train_main(config: TrainConfig) -> None:
    """Main training function."""

    # Print system information
    print_system_info()

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

    (
        train_loader,
        val_loader,
        label_columns,
        original_labels,
        train_labels,
        test_labels
    ) = create_data_loaders(df, image_paths, labels, config)

    trainer = Trainer(
        class_freq=class_frequency,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        label_columns=label_columns,
        init_wandb=True
    )

    # Print trainer information
    trainer.info()

    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    start_time = time.time()
    trainer.train(save_model=True)
    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total training time: {training_time / 60:.2f} minutes")
    print(f"Best validation {trainer.best_model_metric}: {trainer.best_metric_value:.4f}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_type",
        type=str,
        default="action",
        help="Type of model to sweep (e.g., 'action', 'bodyparts')",
    )

    parsed_args = args.parse_args()
    model_to_train = parsed_args.model_type

    train_config = TrainConfig(
        model_type=model_to_train,
    )
    train_main(config=train_config)