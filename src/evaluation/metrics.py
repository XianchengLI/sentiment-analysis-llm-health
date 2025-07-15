"""
Comprehensive evaluation metrics for sentiment analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score
)
import logging

logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """
    Comprehensive evaluator for sentiment analysis performance
    """
    
    def __init__(self, labels: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            labels: List of sentiment labels (default: ['Positive', 'Negative', 'Neutral'])
        """
        self.labels = labels or ['Positive', 'Negative', 'Neutral']
        
    def calculate_metrics(
        self, 
        y_true: List[str], 
        y_pred: List[str],
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_clean = [y_true[i] for i in valid_indices]
        y_pred_clean = [y_pred[i] for i in valid_indices]
        
        if not y_true_clean or not y_pred_clean:
            logger.warning("No valid predictions found")
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_clean, y_pred_clean)
        metrics['precision_macro'] = precision_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true_clean, y_pred_clean, average=None, labels=self.labels, zero_division=0)
        recall_per_class = recall_score(y_true_clean, y_pred_clean, average=None, labels=self.labels, zero_division=0)
        f1_per_class = f1_score(y_true_clean, y_pred_clean, average=None, labels=self.labels, zero_division=0)
        
        for i, label in enumerate(self.labels):
            if i < len(precision_per_class):
                metrics[f'precision_{label.lower()}'] = precision_per_class[i]
                metrics[f'recall_{label.lower()}'] = recall_per_class[i]
                metrics[f'f1_{label.lower()}'] = f1_per_class[i]
        
        # Agreement metrics
        try:
            metrics['cohen_kappa'] = cohen_kappa_score(y_true_clean, y_pred_clean)
        except Exception as e:
            logger.warning(f"Could not calculate Cohen's kappa: {e}")
            metrics['cohen_kappa'] = 0.0
        
        # Response rate
        metrics['response_rate'] = len(y_pred_clean) / len(y_pred)
        
        return metrics
    
    def generate_classification_report(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            str: Formatted classification report
        """
        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        y_true_clean = [y_true[i] for i in valid_indices]
        y_pred_clean = [y_pred[i] for i in valid_indices]
        
        return classification_report(
            y_true_clean, 
            y_pred_clean, 
            labels=self.labels,
            target_names=self.labels,
            zero_division=0
        )