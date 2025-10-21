#!/usr/bin/env python3
"""
Weights & Biases Sweep Runner for ConvNeXt V2 Multilabel Classification
This script runs hyperparameter sweeps using wandb.
"""

import os
import sys
import yaml
import wandb
import subprocess
from pathlib import Path

def run_sweep():
    """Run a wandb sweep for hyperparameter optimization."""
    
    # Load sweep configuration
    with open('wandb_sweep_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config)
    print(f"Sweep created with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/sweeps/{sweep_id}")
    
    # Run the sweep
    wandb.agent(sweep_id, function=train_with_sweep, count=50)  # Run 50 trials

def train_with_sweep():
    """Training function for wandb sweep."""
    
    # Initialize wandb run
    run = wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Create configuration object with sweep parameters
    from config import Config
    
    sweep_config = Config(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        img_size=config.img_size,
        threshold=config.threshold,
        early_stopping_patience=config.early_stopping_patience,
        lr_reduce_patience=config.lr_reduce_patience,
        lr_reduce_factor=config.lr_reduce_factor,
        epochs=config.epochs,
        use_wandb=True,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_tags=config.wandb_tags
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
        train_loader, val_loader, label_columns = create_data_loaders(sweep_config)
        
        # Create model
        model = create_model(sweep_config, num_classes=len(label_columns))
        model = setup_model_for_training(model, sweep_config)
        model = model.to(device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=sweep_config,
            train_loader=train_loader,
            val_loader=val_loader,
            label_columns=label_columns,
            device=device
        )
        
        # Train model
        history = trainer.train()
        
        # Log final metrics
        final_metrics = {
            'final_val_f1_micro': trainer.best_val_f1,
            'final_epochs': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
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
