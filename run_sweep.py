import os
import sys
import yaml
import wandb
import torch

# Import training modules
from training.config import Config
from training.find_batch_size import find_batch_size
from training.utils import compute_class_frequency
from training.dataset import create_data_loaders, load_and_prepare_data
from training.trainer import Trainer


def run_sweep():
    """Run a wandb sweep for hyperparameter optimization."""

    # Load sweep configuration
    with open('wandb_sweep_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Get entity and project from config
    entity = sweep_config.get('wandb_entity', 'miloszbertman')
    project = sweep_config.get('wandb_project', 'fn-content-classifier')
    
    # Initialize wandb sweep
    sweep_id = os.getenv('SWEEP_ID', None)
    
    if not sweep_id:
        sweep_id = input("Input the sweep_id (or press Enter to create new): ").strip()

    if not sweep_id:
        # Create a new sweep
        sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
        print(f"Created new sweep with ID: {sweep_id}")
        
    else:
        print(f"Using existing sweep ID: {sweep_id}")

    # Run the sweep
    wandb.agent(sweep_id, function=train_with_sweep, count=os.getenv('SWEEP_ITERATIONS', 20), entity=entity, project=project)  # Run 50 trials

def train_with_sweep():
    """Training function for wandb sweep."""

    # Get hyperparameters from wandb
    # Initialize wandb run with config values
    wandb.init(
        project="fn-content-classifier",
        entity="miloszbertman",
    )

    config = wandb.config
    sweep_config = Config(
        model_name="convnextv2_tiny",
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        bce_power=config.bce_power,
        tau_logit_adjust=config.tau_logit_adjust,
        weight_decay=config.weight_decay,
        use_wandb=True,
        wandb_project="fn-content-classifier",
        wandb_entity="miloszbertman",
        wandb_tags=["sweep", "multilabel", "convnextv2"]
    )

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df, image_paths, labels = load_and_prepare_data(config=sweep_config)
    class_frequency = compute_class_frequency(df.drop(["file_name"], axis=1))
    sweep_config.batch_size, sweep_config.grad_accum_steps = find_batch_size(
        config=sweep_config,
        num_classes=len(labels),
        device=str(device),
    )

    (
        train_loader,
        val_loader,
        label_columns,
        original_labels,
        train_labels,
        test_labels
    ) = create_data_loaders(df, image_paths, labels, sweep_config)

    print("\nInitializing trainer...")
    trainer = Trainer(
        class_freq=class_frequency,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        label_columns=label_columns,
        device=str(device),
        init_wandb=False,
    )

    # Train model
    trainer.train(save_model=False)

    print(f"Training completed! Best {trainer.best_model_metric}: {trainer.best_metric_value:.4f}")

if __name__ == "__main__":
    # Check if wandb is logged in
    if not wandb.api.api_key:
        print("Please run 'wandb login' first!")
        sys.exit(1)

    # Run the sweep
    run_sweep()
