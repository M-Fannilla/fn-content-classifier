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
        # Create new sweep
        sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
        print(f"New sweep created with ID: {sweep_id}")
    else:
        # Validate existing sweep
        try:
            # Try to get sweep info to validate it exists
            api = wandb.Api()
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            print(f"Using existing sweep with ID: {sweep_id}")
        except Exception as e:
            print(f"Error accessing sweep {sweep_id}: {e}")
            print("Creating new sweep instead...")
            sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
            print(f"New sweep created with ID: {sweep_id}")

    print(f"Sweep URL: https://wandb.ai/sweeps/{sweep_id}")

    # Run the sweep
    wandb.agent(sweep_id, function=train_with_sweep, count=50)  # Run 50 trials

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
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        img_size=config.img_size,
        class_weight_method=config.class_weight_method,
        loss_type=config.loss_type,
        use_wandb=True,
        wandb_project="fn-content-classifier",
        wandb_entity="miloszbertman",
        wandb_tags=["sweep", "multilabel", "convnextv2"]
    )

    # Import training modules
    from dataset import create_data_loaders
    from model import create_model, setup_model_for_training
    from trainer import Trainer
    import torch

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Create data loaders
        train_loader, val_loader, label_columns, _, _, _ = create_data_loaders(sweep_config)

        # Create model
        model = create_model(sweep_config, num_classes=len(label_columns))
        model = setup_model_for_training(model, device=device)
        model = model.to(device)

        # Create trainer
        trainer = Trainer(
            model=model,
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
            'final_val_f1_micro': trainer.best_val_f1,
            'final_epochs': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'class_weight_method': config.class_weight_method,
            'loss_type': config.loss_type
        }

        wandb.log(final_metrics)

        print(f"Training completed! Best F1 Micro: {trainer.best_val_f1:.4f}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        wandb.log({"error": str(e)})
        raise e

if __name__ == "__main__":
    # Check if wandb is logged in
    if not wandb.api.api_key:
        print("Please run 'wandb login' first!")
        sys.exit(1)

    # Run the sweep
    run_sweep()
