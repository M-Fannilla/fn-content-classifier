import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Find optimal thresholds for each class using F1 score optimization.
    
    Args:
        y_true: Ground truth binary labels (n_samples, n_classes)
        y_prob: Predicted probabilities (n_samples, n_classes)
    
    Returns:
        Array of optimal thresholds for each class (n_classes,)
    """
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thr, best_f1 = 0.5, 0
        for thr in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(y_true[:, i], (y_prob[:, i] > thr).astype(int))
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds.append(best_thr)
    return np.array(thresholds)


class MultilabelMetrics:
    """Class for computing multilabel classification metrics."""
    
    def __init__(self, threshold: float | np.ndarray = 0.5):
        self.threshold = threshold
        
    def compute_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_pred_proba: np.ndarray = None
    ) -> dict[str, float]:
        """Compute comprehensive metrics for multilabel classification."""
        
        metrics = {}
        
        # Convert predictions to binary using threshold
        y_pred_binary = (y_pred >= self.threshold).astype(np.int32)
        
        # Micro-averaged metrics
        metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
        
        # Macro-averaged metrics
        metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
        
        # Sample-wise metrics
        metrics['f1_samples'] = f1_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metrics['precision_samples'] = precision_score(y_true, y_pred_binary, average='samples', zero_division=0)
        metrics['recall_samples'] = recall_score(y_true, y_pred_binary, average='samples', zero_division=0)
        
        # AUC metrics (if probabilities are provided)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred_proba, average='micro')
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
                metrics['roc_auc_samples'] = roc_auc_score(y_true, y_pred_proba, average='samples')
                
                metrics['pr_auc_micro'] = average_precision_score(y_true, y_pred_proba, average='micro')
                metrics['pr_auc_macro'] = average_precision_score(y_true, y_pred_proba, average='macro')
                metrics['pr_auc_samples'] = average_precision_score(y_true, y_pred_proba, average='samples')
            except ValueError:
                # Handle case where some classes have no positive samples
                metrics['roc_auc_micro'] = 0.0
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_samples'] = 0.0
                metrics['pr_auc_micro'] = 0.0
                metrics['pr_auc_macro'] = 0.0
                metrics['pr_auc_samples'] = 0.0
        
        return metrics
    
    def compute_per_class_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            class_names: list[str] = None
    ) -> dict[str, dict[str, float]]:
        """Compute per-class metrics."""
        
        y_pred_binary = (y_pred >= self.threshold).astype(np.int32)
        num_classes = y_true.shape[1]
        
        per_class_metrics = {}
        
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f'class_{i}'
            
            # Check if class has any positive samples
            if y_true[:, i].sum() == 0:
                per_class_metrics[class_name] = {
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'support': 0
                }
                continue
            
            f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            precision = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            support = y_true[:, i].sum()
            
            per_class_metrics[class_name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'support': support
            }
        
        return per_class_metrics


def plot_roc_curves(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: list[str],
        save_path: str = None
):
    """Plot ROC curves for each class."""
    
    num_classes = y_true.shape[1]
    fig, axes = plt.subplots(2, (num_classes + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if num_classes > 1 else [axes]
    
    for i in range(num_classes):
        if i >= len(axes):
            break
            
        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
        
        # Check if class has positive samples
        if y_true[:, i].sum() == 0:
            axes[i].text(0.5, 0.5, f'{class_name}\nNo positive samples', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{class_name}')
            continue
        
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        
        axes[i].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{class_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curves(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: list[str],
        save_path: str = None
):
    """Plot Precision-Recall curves for each class."""
    
    num_classes = y_true.shape[1]
    fig, axes = plt.subplots(2, (num_classes + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if num_classes > 1 else [axes]
    
    for i in range(num_classes):
        if i >= len(axes):
            break
            
        class_name = class_names[i] if i < len(class_names) else f'Class {i}'
        
        # Check if class has positive samples
        if y_true[:, i].sum() == 0:
            axes[i].text(0.5, 0.5, f'{class_name}\nNo positive samples', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{class_name}')
            continue
        
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
        
        axes[i].plot(recall, precision, label=f'AP = {ap:.3f}')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].set_title(f'{class_name}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
