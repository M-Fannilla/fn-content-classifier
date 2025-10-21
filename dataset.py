import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import List, Tuple, Dict, Any
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from config import Config


class MultilabelImageDataset(Dataset):
    """Dataset for multilabel image classification with stratified splitting."""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: np.ndarray, 
                 transform: T.Compose = None,
                 config: Config = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (self.config.img_size, self.config.img_size), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.FloatTensor(label)


def get_transforms(config: Config, is_training: bool = True) -> T.Compose:
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


def load_and_prepare_data(config: Config) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """Load and prepare the dataset with stratified splitting."""
    # Load labels
    df = pd.read_csv(config.label_dataframe, index_col=0)

    # Get label columns (all except file_name)
    label_columns = [col for col in df.columns if col != 'file_name']
    labels = df[label_columns].values.astype(np.float32)
    
    # Create image paths
    image_paths = [os.path.join(config.dataset_src, filename) for filename in df['file_name']]
    
    return df, image_paths, labels


def create_stratified_splits(image_paths: List[str], 
                           labels: np.ndarray, 
                           config: Config) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Create stratified train/test splits for multilabel data."""
    
    # Use MultilabelStratifiedKFold for proper stratification
    mlsf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    
    # Get the first split for train/test
    train_idx, test_idx = next(mlsf.split(image_paths, labels))
    
    train_paths = [image_paths[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    return train_paths, test_paths, train_labels, test_labels


def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train and validation data loaders."""
    # Load data
    df, image_paths, labels = load_and_prepare_data(config)
    
    # Create stratified splits
    train_paths, test_paths, train_labels, test_labels = create_stratified_splits(
        image_paths, labels, config
    )
    
    # Get label columns for later use
    label_columns = [col for col in df.columns if col != 'file_name']
    
    # Create transforms
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # Create datasets
    train_dataset = MultilabelImageDataset(train_paths, train_labels, train_transform, config)
    val_dataset = MultilabelImageDataset(test_paths, test_labels, val_transform, config)
    
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
    
    return train_loader, val_loader, label_columns
