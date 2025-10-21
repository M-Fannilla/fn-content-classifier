import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import timm
from tqdm.auto import tqdm

# ->>>> conda activate torch311
# --------------------
# Configuration
# --------------------
DATASET_SIZE_PERC = 0.5
TRAIN_SIZE_PERC = 0.75
VALIDATION_SIZE_PERC = 0.2
TEST_SIZE_PERC = 0.05

DATASET_SRC = './fn-content-dataset/images'
LABEL_DATAFRAME = './fn-content-dataset/action_labels.csv'
OUTPUT_DIR = './outputs'
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4
# Windows compatibility: use 0 workers to avoid shared memory issues
NUM_WORKERS = 0 if os.name == 'nt' else os.cpu_count() // 4
THRESHOLD = 0.5
IMG_SIZE = 224
MODEL_NAME = 'convnextv2_tiny'  # default smaller, can be overridden
CACHE_DECODED_IMAGES = False  # disabled due to memory constraints with large datasets

# Training mode: 'finetune' or 'full_retrain'
TRAINING_MODE = 'finetune'  # default to finetuning
# Finetuning specific settings
FINETUNE_LEARNING_RATE = 1e-5  # lower learning rate for finetuning
FINETUNE_EPOCHS = 20  # fewer epochs for finetuning
# Full retrain specific settings
RETRAIN_LEARNING_RATE = 2e-4  # higher learning rate for full retrain
RETRAIN_EPOCHS = 50  # more epochs for full retrain

# Catalog of ConvNeXt V2 variants and their recommended input sizes
MODEL_CATALOG: Dict[str, int] = {
    'convnextv2_nano': IMG_SIZE,
    'convnextv2_tiny': IMG_SIZE,
    'convnextv2_base': IMG_SIZE,
    'convnextv2_large': IMG_SIZE,
    'convnextv2_huge': IMG_SIZE,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def build_filepaths(df: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    df = df.copy()
    if 'file_name' not in df.columns:
        raise ValueError("Expected 'file_name' column in dataframe")
    df['filepath'] = df['file_name'].apply(lambda x: os.path.join(base_dir, x))
    return df


def select_subset(df: pd.DataFrame, frac: float, label_cols: List[str]) -> pd.DataFrame:
    """
    Select a stratified subset of the dataset maintaining label distribution.
    
    Args:
        df: Input dataframe
        frac: Fraction of data to select (0.0 to 1.0)
        label_cols: List of label column names for stratification
    
    Returns:
        Stratified subset of the dataframe
    """
    if frac >= 1.0:
        return df
    
    # Use MultilabelStratifiedKFold to ensure stratified sampling
    X = df[['filepath']]  # placeholder features
    y = df[label_cols].values
    
    # Calculate number of splits needed to get approximately the desired fraction
    n_splits = max(2, int(1.0 / frac))
    
    # Use stratified k-fold to get one fold as our subset
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # Get the first fold as our subset
    subset_indices, _ = next(mskf.split(X, y))
    
    return df.iloc[subset_indices].reset_index(drop=True)


def get_label_columns(df: pd.DataFrame) -> List[str]:
    non_label_cols = {'file_name', 'filepath'}
    return [c for c in df.columns if c not in non_label_cols]


def plot_label_distribution(df: pd.DataFrame, label_cols: List[str], out_path: str) -> None:
    counts = df[label_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, max(4, len(label_cols) * 0.3)))
    counts.plot(kind='bar')
    plt.title('Label Distribution (counts)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


class MultiLabelImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols: List[str], transform: T.Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fp = row['filepath']
        if CACHE_DECODED_IMAGES and fp in self._cache:
            # cached as numpy array HWC uint8
            np_img = self._cache[fp]
            image = Image.fromarray(np_img)
        else:
            image = Image.open(fp).convert('RGB')
            if CACHE_DECODED_IMAGES:
                self._cache[fp] = np.array(image)
        image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return image, labels


def make_transforms(img_size: int) -> Tuple[T.Compose, T.Compose]:
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tfms, val_tfms


def multilabel_stratified_split(df: pd.DataFrame, label_cols: List[str],
                                train_size: float, val_size: float, test_size: float,
                                seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-6)

    X = df[['filepath']]  # placeholder
    y = df[label_cols].values

    # First split: train_val vs test
    mskf = MultilabelStratifiedKFold(n_splits=int(1 / test_size), shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(mskf.split(X, y))
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Recompute proportions within train_val
    tv_total = len(df_train_val)
    desired_train = int(round(tv_total * (train_size / (train_size + val_size))))

    # Use mskf again inside train_val to get train vs val
    y_tv = df_train_val[label_cols].values
    # Choose n_splits to approximate the desired split; fall back to 5 if small
    inner_splits = max(3, min(10, tv_total // 64))
    inner = MultilabelStratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    best_split = None
    best_diff = 10**9
    for tr_idx, vl_idx in inner.split(df_train_val[['filepath']], y_tv):
        if abs(len(tr_idx) - desired_train) < best_diff:
            best_diff = abs(len(tr_idx) - desired_train)
            best_split = (tr_idx, vl_idx)

    tr_idx, vl_idx = best_split
    df_train = df_train_val.iloc[tr_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[vl_idx].reset_index(drop=True)
    return df_train, df_val, df_test


def compute_pos_weight(df: pd.DataFrame, label_cols: List[str]) -> torch.Tensor:
    counts = df[label_cols].sum().values.astype(np.float32)
    totals = np.array([len(df)] * len(label_cols), dtype=np.float32)
    # pos_weight = (N - P) / P; clamp to avoid div by zero
    pos_weight = (totals - counts) / np.clip(counts, 1.0, None)
    tw = torch.tensor(pos_weight, dtype=torch.float32)
    # Normalize to have mean 1.0 to stabilize loss scaling across runs
    mean_val = torch.clamp(tw.mean(), min=1e-6)
    tw = tw / mean_val
    return tw


def build_model(num_classes: int, training_mode: str = 'finetune') -> nn.Module:
    try:
        if training_mode == 'finetune':
            # For finetuning: use pretrained weights and freeze early layers
            model = timm.create_model(MODEL_NAME, pretrained=True, in_chans=3, num_classes=num_classes)
            
            # Freeze early layers for finetuning
            for name, param in model.named_parameters():
                if 'head' not in name and 'classifier' not in name and 'fc' not in name:
                    param.requires_grad = False
            
            print(f"Finetuning mode: Frozen {sum(1 for p in model.parameters() if not p.requires_grad)} parameters")
            print(f"Finetuning mode: Trainable {sum(1 for p in model.parameters() if p.requires_grad)} parameters")
            
        elif training_mode == 'full_retrain':
            # For full retraining: start from scratch (no pretrained weights)
            model = timm.create_model(MODEL_NAME, pretrained=False, in_chans=3, num_classes=num_classes)
            print(f"Full retrain mode: All {sum(1 for p in model.parameters() if p.requires_grad)} parameters are trainable")
            
        else:
            raise ValueError(f"Invalid training_mode: {training_mode}. Must be 'finetune' or 'full_retrain'")
            
        return model
    except RuntimeError as e:
        available = timm.list_models('convnextv2*')
        raise RuntimeError(f"{e}. Available ConvNeXt V2 models: {available}") from e


@dataclass
class TrainResult:
    train_losses: List[float]
    val_losses: List[float]
    val_aurocs: List[float]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> Tuple[float, float, float]:
    model.eval()
    all_targets, all_logits = [], []
    val_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item() * images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = sigmoid_np(all_logits)

    try:
        auroc = roc_auc_score(all_targets, all_probs, average='macro')
    except Exception:
        auroc = float('nan')

    try:
        mAP = average_precision_score(all_targets, all_probs, average='macro')
    except Exception:
        mAP = float('nan')

    preds = (all_probs >= threshold).astype(np.float32)
    acc = (preds == all_targets).mean()

    val_loss = val_loss / len(loader.dataset)
    return val_loss, auroc, mAP


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: torch.device,
          pos_weight: torch.Tensor,
          epochs: int,
          lr: float,
          threshold: float,
          out_dir: str,
          training_mode: str = 'finetune') -> TrainResult:
    # Configure optimizer based on training mode
    if training_mode == 'finetune':
        # For finetuning: only optimize trainable parameters, use lower learning rate
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"Finetuning optimizer: {len(trainable_params)} trainable parameters")
    else:
        # For full retraining: optimize all parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        print(f"Full retrain optimizer: {sum(1 for p in model.parameters())} total parameters")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_metric = -float('inf')
    history_train_loss, history_val_loss, history_val_auroc = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        iterator = train_loader

        iterator = tqdm(train_loader, desc=f"{device.type.upper()}: Epoch {epoch}/{epochs}", leave=False)

        for images, targets in iterator:
            images = images.to(device, non_blocking=(device.type == 'cuda'))
            targets = targets.to(device, non_blocking=(device.type == 'cuda'))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, auroc, mAP = evaluate(model, val_loader, device, threshold)

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_auroc.append(auroc)

        # Save best by AUROC, fallback to mAP if NaN
        score = auroc if not math.isnan(auroc) else (mAP if not math.isnan(mAP) else -float('inf'))
        if score > best_metric:
            best_metric = score
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_auroc': auroc,
                        'val_map': mAP}, os.path.join(out_dir, 'best_model.pt'))

        print(f"Epoch {epoch:02d}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} AUROC: {auroc:.4f} mAP: {mAP:.4f}")

        # Plot curves each epoch
        plot_training_curves(history_train_loss, history_val_loss, history_val_auroc,
                             os.path.join(out_dir, 'training_curves.png'))

    return TrainResult(history_train_loss, history_val_loss, history_val_auroc)


def plot_training_curves(train_losses: List[float], val_losses: List[float], val_aurocs: List[float], out_path: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_aurocs, color='green', label='Val AUROC')
    ax2.set_ylabel('AUROC')
    ax2.legend(loc='lower right')

    plt.title('Training Curves')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataset by checking if image files exist and remove missing ones."""
    temp_df = df.copy()
    missing_indices = []

    def _process_file(index: int, filepath: str) -> int:
        if not os.path.exists(filepath):
            return index
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df), desc="Validating dataset..."):
            futures.append(executor.submit(_process_file, index, row['filepath']))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing validation of dataset..."):
            result = future.result()
            if result is not None:
                missing_indices.append(result)
    
    # Remove missing files
    if missing_indices:
        temp_df = temp_df.drop(missing_indices)
        temp_df.to_csv(LABEL_DATAFRAME)
        print(f"Removed {len(missing_indices)} missing files from dataset")

    return temp_df



def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train ConvNeXt model for multilabel classification')
    
    # Training mode
    parser.add_argument('--mode', type=str, choices=['finetune', 'full_retrain'], 
                       default=TRAINING_MODE,
                       help='Training mode: finetune (default) or full_retrain')
    
    # Model configuration
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       help=f'Model name (default: {MODEL_NAME})')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    
    # Dataset configuration
    parser.add_argument('--dataset_size', type=float, default=DATASET_SIZE_PERC,
                       help=f'Dataset size percentage (default: {DATASET_SIZE_PERC})')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                       help=f'Image size (default: {IMG_SIZE})')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    
    return parser.parse_args()



def main() -> None:
    # Parse command line arguments
    args = parse_args()
    
    # Override global config with command line arguments
    global MODEL_NAME, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, DATASET_SIZE_PERC, IMG_SIZE, OUTPUT_DIR
    MODEL_NAME = args.model
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    DATASET_SIZE_PERC = args.dataset_size
    IMG_SIZE = args.img_size
    OUTPUT_DIR = args.output_dir
    
    # Set training mode specific parameters
    training_mode = args.mode
    if training_mode == 'finetune':
        LEARNING_RATE = FINETUNE_LEARNING_RATE
        NUM_EPOCHS = FINETUNE_EPOCHS
    elif training_mode == 'full_retrain':
        LEARNING_RATE = RETRAIN_LEARNING_RATE
        NUM_EPOCHS = RETRAIN_EPOCHS
    
    set_seed(SEED)
    ensure_output_dir(OUTPUT_DIR)
    import logging
    logger = logging.getLogger("trainer")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(OUTPUT_DIR, 'run.log'))
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.info(f"Seed set to {SEED}")
    logger.info(f"Training mode: {training_mode}")
    # Resolve effective image size based on selected model
    effective_img_size = MODEL_CATALOG.get(MODEL_NAME)
    logger.info(f"Config: batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, model={MODEL_NAME}, img_size={effective_img_size}")

    df = load_dataframe(LABEL_DATAFRAME)
    logger.info(f"Loaded dataframe from {LABEL_DATAFRAME} with shape {df.shape}")
    df = build_filepaths(df, DATASET_SRC)
    # df = validate_dataset(df)
    
    # Get label columns before subset selection for stratified sampling
    label_cols = get_label_columns(df)
    df = select_subset(df, DATASET_SIZE_PERC, label_cols)
    logger.info(f"Using stratified subset fraction={DATASET_SIZE_PERC}; subset size={len(df)}")
    # Save label names
    with open(os.path.join(OUTPUT_DIR, 'labels.json'), 'w') as f:
        json.dump(label_cols, f, indent=2)
    logger.info(f"Detected {len(label_cols)} labels")

    # Plot label distribution
    plot_label_distribution(df, label_cols, os.path.join(OUTPUT_DIR, 'label_distribution.png'))
    logger.info("Saved label distribution plot")

    # Stratified splits
    df_train, df_val, df_test = multilabel_stratified_split(
        df, label_cols, TRAIN_SIZE_PERC, VALIDATION_SIZE_PERC, TEST_SIZE_PERC, SEED
    )
    logger.info(f"Split sizes -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    train_tfms, val_tfms = make_transforms(effective_img_size)
    train_ds = MultiLabelImageDataset(df_train, label_cols, train_tfms)
    val_ds = MultiLabelImageDataset(df_val, label_cols, val_tfms)
    test_ds = MultiLabelImageDataset(df_test, label_cols, val_tfms)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    logger.info(f"Loader sizes -> train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=len(label_cols), training_mode=training_mode).to(device)
    logger.info(f"Using device: {device}{' - ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    logger.info(f"Model {MODEL_NAME} initialized with {sum(p.numel() for p in model.parameters())} params")

    pos_weight = compute_pos_weight(df_train, label_cols)
    logger.info(f"pos_weight stats -> min: {pos_weight.min().item():.3f}, max: {pos_weight.max().item():.3f}")
    result = train(model, train_loader, val_loader, device, pos_weight, NUM_EPOCHS, LEARNING_RATE, THRESHOLD, OUTPUT_DIR, training_mode)

    # Evaluate best model on test set
    ckpt_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    logger.info("Loaded best checkpoint for test evaluation" if os.path.exists(ckpt_path) else "Evaluating current model on test set")
    test_loss, test_auroc, test_mAP = evaluate(model, test_loader, device, THRESHOLD)

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump({'test_loss': test_loss, 'test_auroc': float(test_auroc), 'test_mAP': float(test_mAP)}, f, indent=2)
    logger.info(f"Test - loss: {test_loss:.4f} AUROC: {test_auroc:.4f} mAP: {test_mAP:.4f}")


if __name__ == '__main__':
    main()
