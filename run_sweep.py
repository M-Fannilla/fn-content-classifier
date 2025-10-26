import sys
import yaml
import wandb

def run_sweep():
    """Run a wandb sweep for hyperparameter optimization."""

    # Load sweep configuration
    with open('wandb_sweep_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Get entity and project from config
    entity = sweep_config.get('wandb_entity', 'miloszbertman')
    project = sweep_config.get('wandb_project', 'fn-content-classifier')
    
    # Initialize wandb sweep
    sweep_id = input("Input the sweep_id (or press Enter to create new): ").strip()

    if not sweep_id:
        # Create a new sweep
        sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Using existing sweep ID: {sweep_id}")

    # Run the sweep
    wandb.agent(sweep_id, function=train_with_sweep, count=50, entity=entity, project=project)  # Run 50 trials

def train_with_sweep():
    """Training function for wandb sweep."""

    # Get hyperparameters from wandb
    # Initialize wandb run with config values
    try:
        wandb.init(
            project="fn-content-classifier",
            entity="miloszbertman",
        )
    except Exception as e:
        print(f"Failed to initialize wandb run: {e}")
        # Try without project/entity specification
        wandb.init()
    
    config = wandb.config

    # Create configuration object with sweep parameters
    from config import Config

    sweep_config = Config(
        model_name="convnextv2_tiny",
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        bce_power=config.bce_power,
        tau_logit_adjust=config.tau_logit_adjust,
        weight_decay=config.weight_decay,
        lr_reduce_patience=config.lr_reduce_patience,
        use_wandb=True,
        wandb_project="fn-content-classifier",
        wandb_entity="miloszbertman",
        wandb_tags=["sweep", "multilabel", "convnextv2"]
    )

    # Import training modules
    from dataset import create_data_loaders, load_and_prepare_data
    from model import create_model, setup_model_for_training
    from trainer import Trainer
    import torch
    from utils import compute_class_frequency

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df, image_paths, labels = load_and_prepare_data(config=sweep_config)
    # Create data loaders
    class_frequency = compute_class_frequency(df.drop(["file_name"], axis=1))
    (
        train_loader,
        val_loader,
        label_columns,
        original_labels,
        train_labels,
        test_labels
    ) = create_data_loaders(df, image_paths, labels, sweep_config)

    # Create model
    model = create_model(sweep_config, num_classes=len(label_columns))
    model = setup_model_for_training(
        sweep_config, model, class_freq=class_frequency, device=device
    )
    model = model.to(device)

    trainer = Trainer(
        model=model,
        class_freq=class_frequency,
        config=sweep_config,
        train_loader=train_loader,
        val_loader=val_loader,
        label_columns=label_columns,
        device=str(device)
    )

    # Train model
    history = trainer.train(save_model=False)

    # Log final metrics
    final_metrics = {
        f'final_val_{trainer.best_model_metric}': trainer.best_metric_value,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'learning_rate': sweep_config.learning_rate,
        'batch_size': sweep_config.batch_size,
        'bce_power': sweep_config.bce_power,
        'tau_logit_adjust': sweep_config.tau_logit_adjust,
        'weight_decay': sweep_config.weight_decay,
        'loss_type': config.loss_type
    }

    wandb.log(final_metrics)

    print(f"Training completed! Best {trainer.best_model_metric}: {trainer.best_metric_value:.4f}")

if __name__ == "__main__":
    # Check if wandb is logged in
    if not wandb.api.api_key:
        print("Please run 'wandb login' first!")
        sys.exit(1)

    # Run the sweep
    run_sweep()
