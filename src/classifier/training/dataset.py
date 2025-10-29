import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt

# Import custom modules
from .. import DATASETS_DIR
from ..configs import TrainConfig


class MultilabelImageDataset(Dataset):
    """Dataset for multilabel image classification with stratified splitting."""

    def __init__(
            self,
            image_paths: list[str],
            labels: np.ndarray,
            transform: T.Compose = None,
            config: TrainConfig = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)


def get_transforms(config: TrainConfig, is_training: bool = True) -> T.Compose:
    """Get data transforms for training or validation."""
    if is_training:
        return T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def load_and_prepare_data(
        config: TrainConfig
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load and prepare the dataset with stratified splitting."""
    # Load labels
    df = pd.read_csv(config.label_dataframe, index_col=0)

    # Get label columns (all except file_name)
    label_columns = [col for col in df.columns if col != 'file_name']
    labels = df[label_columns].values.astype(np.float32)

    # Create image paths
    image_paths = [str(DATASETS_DIR / filename) for filename in df['file_name']]

    return df, image_paths, labels


def create_stratified_splits(
        image_paths: list[str],
        labels: np.ndarray,
        config: TrainConfig
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Create stratified train/test splits for multilabel data."""

    # Use MultilabelStratifiedKFold for proper stratification
    mlsf = MultilabelStratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.seed
    )

    # Get the first split for train/test
    train_idx, test_idx = next(mlsf.split(image_paths, labels))

    train_paths = [image_paths[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    return train_paths, test_paths, train_labels, test_labels

def create_data_loaders(
        df, image_paths, labels, config: TrainConfig
) -> tuple[DataLoader, DataLoader, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Create train and validation data loaders.

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        label_columns: List of class names
        original_labels: Original dataset labels
        train_labels: Training dataset labels
        test_labels: Test dataset labels
    """
    # Create stratified splits
    train_paths, test_paths, train_labels, test_labels = create_stratified_splits(
        image_paths=image_paths,
        labels=labels,
        config=config
    )

    # Get label columns for later use
    label_columns = [col for col in df.columns if col != 'file_name']

    # Create transforms
    train_transform = get_transforms(
        config=config,
        is_training=True
    )
    val_transform = get_transforms(
        config=config,
        is_training=False
    )

    # Create datasets
    train_dataset = MultilabelImageDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform,
        config=config
    )
    val_dataset = MultilabelImageDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=val_transform,
        config=config
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print("Dataset loaded successfully!")
    print(f"  Number of classes: {len(label_columns)}")
    print(f"  Class names: {label_columns}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    return (
        train_loader,
        val_loader,
        label_columns,
        labels,
        train_labels,
        test_labels
    )


def plot_label_distribution(
        original_labels: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        label_columns: list[str],
        save_path: str = None
) -> None:
    """Plot label distribution comparison between original, train, and test datasets.

    Args:
        original_labels: Original dataset labels (n_samples, n_classes)
        train_labels: Training dataset labels (n_train, n_classes)
        test_labels: Test dataset labels (n_test, n_classes)
        label_columns: List of class names
        save_path: Optional path to save the plot
    """
    # Calculate positive counts for each class
    original_counts = original_labels.sum(axis=0)
    train_counts = train_labels.sum(axis=0)
    test_counts = test_labels.sum(axis=0)

    # Calculate total samples
    n_original = len(original_labels)
    n_train = len(train_labels)
    n_test = len(test_labels)

    # Calculate percentages
    original_percentages = (original_counts / n_original) * 100
    train_percentages = (train_counts / n_train) * 100
    test_percentages = (test_counts / n_test) * 100

    # Create figure with subplots
    n_classes = len(label_columns)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Absolute counts
    x = np.arange(n_classes)
    width = 0.25

    axes[0].bar(x - width, original_counts, width, label='Original', color='blue', alpha=0.7)
    axes[0].bar(x, train_counts, width, label='Train', color='green', alpha=0.7)
    axes[0].bar(x + width, test_counts, width, label='Test', color='orange', alpha=0.7)

    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Positive Sample Count')
    axes[0].set_title('Label Distribution: Absolute Counts')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(label_columns, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Percentages
    axes[1].bar(x - width, original_percentages, width, label='Original', color='blue', alpha=0.7)
    axes[1].bar(x, train_percentages, width, label='Train', color='green', alpha=0.7)
    axes[1].bar(x + width, test_percentages, width, label='Test', color='orange', alpha=0.7)

    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Positive Sample Percentage (%)')
    axes[1].set_title('Label Distribution: Percentages')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_columns, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add summary statistics
    fig.suptitle(f'Dataset Split Analysis\n'
                 f'Original: {n_original:,} samples | '
                 f'Train: {n_train:,} samples ({n_train/n_original*100:.1f}%) | '
                 f'Test: {n_test:,} samples ({n_test/n_original*100:.1f}%)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Label distribution plot saved to: {save_path}")

    plt.show()

    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED LABEL DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"{'Class':<20} {'Original':<12} {'Train':<12} {'Test':<12} {'Train%':<8} {'Test%':<8}")
    print("-"*80)

    for i, class_name in enumerate(label_columns):
        train_pct = train_percentages[i]
        test_pct = test_percentages[i]

        print(f"{class_name:<20} {original_counts[i]:<12} {train_counts[i]:<12} {test_counts[i]:<12} "
              f"{train_pct:<8.2f} {test_pct:<8.2f}")

    print("-"*80)
    print(f"{'TOTAL':<20} {n_original:<12} {n_train:<12} {n_test:<12} "
          f"{n_train/n_original*100:<8.1f} {n_test/n_original*100:<8.1f}")
    print("="*80)

    # Calculate and print stratification quality
    print("\nSTRATIFICATION QUALITY:")
    print(f"  Average absolute difference (Train vs Original): {np.mean(np.abs(train_percentages - original_percentages)):.2f}%")
    print(f"  Average absolute difference (Test vs Original): {np.mean(np.abs(test_percentages - original_percentages)):.2f}%")
    print(f"  Max absolute difference (Train vs Original): {np.max(np.abs(train_percentages - original_percentages)):.2f}%")
    print(f"  Max absolute difference (Test vs Original): {np.max(np.abs(test_percentages - original_percentages)):.2f}%")

    # Identify most imbalanced classes
    original_imbalance = original_percentages
    most_rare = np.argsort(original_imbalance)[:3]  # 3 most rare classes
    most_common = np.argsort(original_imbalance)[-3:]  # 3 most common classes

    print("\nMOST IMBALANCED CLASSES:")
    print(f"  Most rare: {[label_columns[i] for i in most_rare]} "
          f"({[f'{original_imbalance[i]:.2f}%' for i in most_rare]})")
    print(f"  Most common: {[label_columns[i] for i in most_common]} "
          f"({[f'{original_imbalance[i]:.2f}%' for i in most_common]})")
