import os
import sys
import yaml
import wandb

# Import training modules
from configs import TrainConfig
from training.utils import set_seed
from .training.utils import compute_class_frequency
from .training.dataset import create_data_loaders, load_and_prepare_data
from .training.trainer import Trainer

WANDB_ENTITY: str | None = None
WANDB_PROJECT: str | None = None
SWEEP_ITERATIONS: int = int(os.getenv('SWEEP_ITERATIONS', 20))
SWEEP_ID: str | None = os.getenv('SWEEP_ID', None)

def run_sweep(model_type: str):
    """Run a wandb sweep for hyperparameter optimization."""

    global WANDB_ENTITY, WANDB_PROJECT, SWEEP_ID

    # Load sweep configuration
    with open(f'sweep_{model_type}.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Get entity and project from config
    WANDB_ENTITY = sweep_config.get('wandb_entity')
    WANDB_PROJECT = sweep_config.get('wandb_project')
    
    # Initialize wandb sweep
    if not SWEEP_ID:
        SWEEP_ID = input("Input the sweep_id (or press Enter to create new): ").strip()

    # Create a new sweep
    if not SWEEP_ID:
        SWEEP_ID = wandb.sweep(sweep_config, entity=WANDB_ENTITY, project=WANDB_PROJECT)
        print(f"Created new sweep with ID: {SWEEP_ID}")
    else:
        print(f"Using existing sweep ID: {SWEEP_ID}")

    # Run the sweep
    wandb.agent(
        SWEEP_ID,
        function=train_with_sweep,
        count=SWEEP_ITERATIONS,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

def train_with_sweep():
    """Training function for wandb sweep."""
    global WANDB_ENTITY, WANDB_PROJECT
    # Initialize wandb run with config values
    wandb.init(
        project="fn-content-classifier",
        entity="miloszbertman",
    )

    # Get hyperparameters from wandb
    config = wandb.config
    sweep_config = TrainConfig(
        model_name="convnextv2_tiny",
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        bce_power=config.bce_power,
        tau_logit_adjust=config.tau_logit_adjust,
        weight_decay=config.weight_decay,
        use_wandb=True,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        wandb_tags=["sweep", "multilabel", "convnextv2"],
    )
    set_seed(sweep_config.seed)

    df, image_paths, labels = load_and_prepare_data(config=sweep_config)
    class_frequency = compute_class_frequency(df.drop(["file_name"], axis=1))

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
        config=sweep_config,
        train_loader=train_loader,
        val_loader=val_loader,
        label_columns=label_columns,
        init_wandb=False,
    )

    # Train model
    trainer.train(save_model=False)

    print(f"Training completed! Best {trainer.best_model_metric}: {trainer.best_metric_value:.4f}")

if __name__ == "__main__":
    model_type = "action"
    # Check if wandb is logged in
    if not wandb.api.api_key:
        print("Please run 'wandb login' first!")
        sys.exit(1)

    # Run the sweep
    run_sweep(model_type=model_type)
