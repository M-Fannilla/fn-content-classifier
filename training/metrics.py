import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score,
    precision_score,
    recall_score
)



class MultilabelMetrics:
    """Class for computing multilabel classification metrics."""

    @staticmethod
    def compute_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            threshold: float | np.ndarray,
            y_pred_proba: np.ndarray = None,
    ) -> dict[str, float]:
        """Compute comprehensive metrics for multilabel classification."""
        metrics = {}
        
        # Convert predictions to binary using threshold
        y_pred_binary = (y_pred >= threshold).astype(np.int32)
        
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
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
            metrics['roc_auc_samples'] = roc_auc_score(y_true, y_pred_proba, average='samples')

            metrics['pr_auc_micro'] = average_precision_score(y_true, y_pred_proba, average='micro')
            metrics['pr_auc_macro'] = average_precision_score(y_true, y_pred_proba, average='macro')
            metrics['pr_auc_samples'] = average_precision_score(y_true, y_pred_proba, average='samples')
        
        return metrics

    @staticmethod
    def compute_per_class_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            threshold: float | np.ndarray,
            class_names: list[str] = None
    ) -> dict[str, dict[str, float]]:
        """Compute per-class metrics."""
        y_pred_binary = (y_pred >= threshold).astype(np.int32)
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
                'support': int(support)
            }
        
        return per_class_metrics