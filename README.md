# ConvNeXt V2 Multilabel Classification Training

This project provides a comprehensive training system for ConvNeXt V2 models on multilabel image classification tasks with proper handling of class imbalance.

## Features

- **ConvNeXt V2 Models**: Support for nano, tiny, base, large, and huge variants
- **Multilabel Classification**: Proper handling of multiple labels per image
- **Class Imbalance Handling**: Stratified splitting and specialized loss functions
- **Mixed Precision Training**: GPU acceleration with automatic mixed precision
- **Early Stopping**: Prevents overfitting by stopping when no improvement
- **Learning Rate Reduction**: Automatically reduces LR when performance plateaus
- **Comprehensive Metrics**: F1, Precision, Recall, ROC AUC, PR AUC
- **Visualization**: ROC curves, Precision-Recall curves, training history
- **Finetuning Only**: Fast training with frozen backbone and trainable classifier
- **Weights & Biases Integration**: Hyperparameter sweeping and experiment tracking

## Installation

```bash
pip install -r requirements.txt
wandb login  # Login to Weights & Biases
```

## Usage

### 1. Prepare Your Dataset

Your dataset should have:
- Images in a directory (e.g., `./compiled/compiled/`)
- A CSV file with labels (e.g., `./compiled/compiled/action_labels.csv`)
- CSV format: `file_name` column + one-hot encoded label columns

Example CSV structure:
```csv
file_name,class1,class2,class3,class4
image1.jpg,1,0,1,0
image2.jpg,0,1,0,1
image3.jpg,1,1,0,0
```

### 2. Configure Training

Edit the configuration in `model_train.ipynb` cell 2:

```python
config = Config(
    dataset_src='./compiled/compiled',  # Path to images
    label_dataframe='./compiled/compiled/action_labels.csv',  # Path to labels
    output_dir='./outputs',  # Where to save results
    model_name='convnextv2_tiny',  # Model variant
    learning_rate=1e-5,  # Learning rate for finetuning
    epochs=20,  # Number of epochs
    batch_size=32,
    img_size=224,
    # ... other parameters
    # Early stopping settings
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    # Learning rate reduction on plateau
    lr_reduce_patience=5,
    lr_reduce_factor=0.5,
    lr_reduce_min_lr=1e-7
)
```

### 3. Run Training

#### Option A: Single Training Run
Execute the cells in `model_train.ipynb` in order:

1. **Setup**: Login to wandb and import modules
2. **Configuration**: Set training parameters and enable wandb
3. **Data Loading**: Load and prepare dataset with stratified splitting
4. **Model Creation**: Create and setup ConvNeXt V2 model
5. **Training Setup**: Initialize trainer with loss function and optimizer
6. **Training**: Execute training loop with mixed precision and wandb logging
7. **Evaluation**: Compute comprehensive metrics
8. **Visualization**: Generate ROC and PR curves
9. **Saving**: Save model, history, and configuration

#### Option B: Hyperparameter Sweep (Recommended)
For finding the best hyperparameters automatically:

```bash
# Run automated hyperparameter sweep
python run_sweep.py

# Or create a sweep and run manually
wandb sweep wandb_sweep_config.yaml
wandb agent <sweep_id>
```
5. **Training Setup**: Initialize trainer with loss function and optimizer
6. **Training**: Execute training loop with mixed precision
7. **Evaluation**: Compute comprehensive metrics
8. **Visualization**: Generate ROC and PR curves
9. **Saving**: Save model, history, and configuration

## Model Variants

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| convnextv2_nano | ~3.7M | Fastest | Good |
| convnextv2_tiny | ~28M | Fast | Better |
| convnextv2_base | ~88M | Medium | Good |
| convnextv2_large | ~198M | Slow | Better |
| convnextv2_huge | ~660M | Slowest | Best |

## Training Mode

### Finetuning
- **Method**: Freezes backbone, trains only classifier head
- **Learning Rate**: 1e-5 (optimized for finetuning)
- **Epochs**: 20 (with early stopping)
- **Benefits**: Fast training, good for small to medium datasets, preserves pretrained features
- **Parameters**: Only ~300K trainable parameters (vs ~28M total)

## Training Features

### Early Stopping
- **Purpose**: Prevents overfitting by stopping when validation performance stops improving
- **Monitoring**: Validation F1 Micro score
- **Patience**: Number of epochs to wait before stopping (default: 10)
- **Min Delta**: Minimum improvement required to reset patience (default: 0.001)
- **Benefit**: Saves training time and prevents overfitting

### Learning Rate Reduction on Plateau
- **Purpose**: Automatically reduces learning rate when performance plateaus
- **Monitoring**: Validation F1 Micro score
- **Patience**: Number of epochs to wait before reducing LR (default: 5)
- **Factor**: Factor by which to reduce LR (default: 0.5)
- **Min LR**: Minimum learning rate threshold (default: 1e-7)
- **Benefit**: Better convergence and final performance

## Weights & Biases Integration

### Hyperparameter Sweeping
The system includes comprehensive wandb integration for hyperparameter optimization:

#### Sweep Configuration
- **Models**: convnextv2_nano, convnextv2_tiny, convnextv2_base
- **Learning Rate**: 1e-6 to 1e-3 (log scale)
- **Batch Size**: 16, 32, 64
- **Image Size**: 224, 256, 384
- **Threshold**: 0.3, 0.4, 0.5, 0.6, 0.7
- **Early Stopping**: 5, 10, 15 epochs patience
- **LR Reduction**: 3, 5, 7 epochs patience with 0.3, 0.5, 0.7 factors

#### Tracked Metrics
- **Primary**: val_f1_micro (optimization target)
- **Secondary**: val_f1_macro, val_f1_samples
- **Precision/Recall**: Micro and macro averages
- **AUC Metrics**: ROC AUC and PR AUC (micro/macro)
- **Training**: Loss curves, learning rate schedule
- **System**: GPU utilization, memory usage

#### Usage
```bash
# Run automated sweep
python run_sweep.py

# Manual sweep creation
wandb sweep wandb_sweep_config.yaml
wandb agent <sweep_id>
```

### Experiment Tracking
- **Automatic Logging**: All metrics logged to wandb dashboard
- **Model Architecture**: Automatic model graph visualization
- **Hyperparameters**: Complete configuration tracking
- **Artifacts**: Model checkpoints and training history
- **Visualizations**: ROC curves, PR curves, training plots

## Loss Functions

The system includes several loss functions for handling class imbalance:

- **Focal Loss**: Default, good for imbalanced data
- **Asymmetric Loss**: Alternative for imbalanced data
- **Weighted BCE**: Simple weighted approach
- **Standard BCE**: Basic binary cross-entropy

## Metrics

The system computes comprehensive metrics:

- **F1 Score**: Micro, macro, and sample-wise
- **Precision/Recall**: Micro, macro, and sample-wise
- **ROC AUC**: Micro, macro, and sample-wise
- **PR AUC**: Micro, macro, and sample-wise
- **Per-class metrics**: Individual class performance

## Output Files

After training, the following files are saved to the output directory:

- `{model_name}_{training_mode}_best.pth`: Trained model
- `training_history.csv`: Training metrics over time
- `training_history.png`: Training curves plot
- `roc_curves.png`: ROC curves for each class
- `pr_curves.png`: Precision-Recall curves for each class
- `config.json`: Training configuration

## Customization

### Changing Models
```python
config.model_name = 'convnextv2_base'  # Change model variant
```

### Changing Learning Rate
```python
config.learning_rate = 2e-5  # Adjust learning rate
```

### Changing Loss Function
Edit `trainer.py` line 45:
```python
self.criterion = get_loss_function('asymmetric', device=device)
```

### Changing Hyperparameters
```python
config.batch_size = 64
config.img_size = 256
config.learning_rate = 2e-5
config.epochs = 30
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space for models and outputs

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `config.batch_size = 16`
- Use smaller model: `config.model_name = 'convnextv2_nano'`
- Reduce image size: `config.img_size = 192`

### Slow Training
- Increase batch size: `config.batch_size = 64`
- Use smaller model: `config.model_name = 'convnextv2_tiny'`
- Reduce image size: `config.img_size = 192`

### Poor Performance
- Use larger model: `config.model_name = 'convnextv2_base'`
- Increase epochs: `config.epochs = 30`
- Adjust learning rate: `config.learning_rate = 2e-5`
- Adjust threshold: `config.threshold = 0.3`

## License

This project is open source and available under the MIT License.
