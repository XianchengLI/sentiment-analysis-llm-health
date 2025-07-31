"""
Confidence Calibration Analysis Module
Provides tools for evaluating and visualizing confidence calibration in LLM predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalibrationAnalyzer:
    """
    Analyzer for confidence calibration in sentiment predictions
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize confidence calibration analyzer
        
        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins
        
    def calculate_calibration_metrics(
        self, 
        confidence_scores: List[float], 
        predictions: List[str],
        true_labels: List[str]
    ) -> Dict[str, float]:
        """
        Calculate confidence calibration metrics
        
        Args:
            confidence_scores: List of confidence scores (0-1)
            predictions: List of predicted labels
            true_labels: List of true labels
            
        Returns:
            Dict[str, float]: Calibration metrics
        """
        # Filter valid predictions
        valid_indices = [
            i for i, (conf, pred) in enumerate(zip(confidence_scores, predictions))
            if conf is not None and pred is not None
        ]
        
        if not valid_indices:
            logger.warning("No valid confidence scores found")
            return {}
        
        conf_clean = [confidence_scores[i] for i in valid_indices]
        pred_clean = [predictions[i] for i in valid_indices]
        true_clean = [true_labels[i] for i in valid_indices]
        
        # Convert to binary accuracy (correct/incorrect)
        accuracy_binary = [1 if pred == true else 0 for pred, true in zip(pred_clean, true_clean)]
        
        metrics = {}
        
        try:
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                accuracy_binary, conf_clean, n_bins=self.n_bins, strategy='uniform'
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in conf_clean]
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean([accuracy_binary[i] for i, in_b in enumerate(in_bin) if in_b])
                    avg_confidence_in_bin = np.mean([conf_clean[i] for i, in_b in enumerate(in_bin) if in_b])
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['expected_calibration_error'] = ece
            
            # Brier Score
            metrics['brier_score'] = brier_score_loss(accuracy_binary, conf_clean)
            
            # Reliability (correlation between confidence and accuracy)
            if len(conf_clean) > 1:
                metrics['confidence_accuracy_correlation'] = np.corrcoef(conf_clean, accuracy_binary)[0, 1]
            else:
                metrics['confidence_accuracy_correlation'] = 0.0
            
            # Confidence statistics
            metrics['mean_confidence'] = np.mean(conf_clean)
            metrics['std_confidence'] = np.std(conf_clean)
            metrics['min_confidence'] = np.min(conf_clean)
            metrics['max_confidence'] = np.max(conf_clean)
            
        except Exception as e:
            logger.error(f"Error calculating calibration metrics: {e}")
            
        return metrics
    
    def generate_calibration_data(
        self, 
        confidence_scores: List[float], 
        predictions: List[str],
        true_labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Generate data for calibration plots
        
        Args:
            confidence_scores: List of confidence scores
            predictions: List of predicted labels  
            true_labels: List of true labels
            
        Returns:
            Tuple of (mean_predicted_values, fraction_of_positives, confidence_clean)
        """
        # Filter valid predictions
        valid_indices = [
            i for i, (conf, pred) in enumerate(zip(confidence_scores, predictions))
            if conf is not None and pred is not None
        ]
        
        conf_clean = [confidence_scores[i] for i in valid_indices]
        pred_clean = [predictions[i] for i in valid_indices]
        true_clean = [true_labels[i] for i in valid_indices]
        
        # Convert to binary accuracy
        accuracy_binary = [1 if pred == true else 0 for pred, true in zip(pred_clean, true_clean)]
        
        # Generate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracy_binary, conf_clean, n_bins=self.n_bins, strategy='uniform'
        )
        
        return mean_predicted_value, fraction_of_positives, conf_clean


class ConfidenceVisualizer:
    """
    Visualizer for confidence calibration analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Initialize visualizer
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        
    def plot_calibration_curve(
        self, 
        confidence_scores: List[float], 
        predictions: List[str],
        true_labels: List[str],
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confidence calibration curve and distribution
        
        Args:
            confidence_scores: List of confidence scores
            predictions: List of predicted labels
            true_labels: List of true labels
            model_name: Name of the model for plot title
            save_path: Path to save the plot
            
        Returns:
            matplotlib.Figure: The generated figure
        """
        analyzer = ConfidenceCalibrationAnalyzer()
        mean_predicted, fraction_positive, conf_clean = analyzer.generate_calibration_data(
            confidence_scores, predictions, true_labels
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
        ax1.plot(mean_predicted, fraction_positive, 'o-', label=f'{model_name}', linewidth=2, markersize=6)
        ax1.set_xlabel('Mean Predicted Confidence')
        ax1.set_ylabel('Fraction of Correct Predictions')
        ax1.set_title(f'Calibration Curve - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Confidence distribution
        ax2.hist(conf_clean, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Confidence Distribution - {model_name}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_models_calibration(
        self,
        models_data: Dict[str, Dict[str, List]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot calibration curves for multiple models
        
        Args:
            models_data: Dict with model names as keys and data dicts as values
                        Data dict should contain 'confidence', 'predictions', 'true_labels'
            save_path: Path to save the plot
            
        Returns:
            matplotlib.Figure: The generated figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        analyzer = ConfidenceCalibrationAnalyzer()
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_data)))
        
        # Plot perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7, linewidth=2)
        
        for i, (model_name, data) in enumerate(models_data.items()):
            try:
                mean_predicted, fraction_positive, conf_clean = analyzer.generate_calibration_data(
                    data['confidence'], data['predictions'], data['true_labels']
                )
                
                # Calibration curve
                ax1.plot(mean_predicted, fraction_positive, 'o-', 
                        label=model_name, color=colors[i], linewidth=2, markersize=6)
                
                # Confidence distribution
                ax2.hist(conf_clean, bins=20, alpha=0.6, color=colors[i], 
                        label=model_name, edgecolor='black')
                
            except Exception as e:
                logger.warning(f"Could not plot data for {model_name}: {e}")
        
        # Customize calibration plot
        ax1.set_xlabel('Mean Predicted Confidence')
        ax1.set_ylabel('Fraction of Correct Predictions')
        ax1.set_title('Calibration Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Customize distribution plot
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distributions Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def analyze_confidence_calibration(
    results_df: pd.DataFrame,
    confidence_col: str = 'Confidence',
    prediction_col: str = 'Sentiment', 
    true_label_col: str = 'True_Sentiment',
    model_col: str = 'Model'
) -> pd.DataFrame:
    """
    Analyze confidence calibration for multiple models from a results DataFrame
    
    Args:
        results_df: DataFrame containing results
        confidence_col: Column name for confidence scores
        prediction_col: Column name for predictions
        true_label_col: Column name for true labels  
        model_col: Column name for model names
        
    Returns:
        pd.DataFrame: Calibration metrics for each model
    """
    analyzer = ConfidenceCalibrationAnalyzer()
    calibration_results = []
    
    for model in results_df[model_col].unique():
        model_data = results_df[results_df[model_col] == model]
        
        metrics = analyzer.calculate_calibration_metrics(
            confidence_scores=model_data[confidence_col].tolist(),
            predictions=model_data[prediction_col].tolist(),
            true_labels=model_data[true_label_col].tolist()
        )
        
        metrics['model'] = model
        calibration_results.append(metrics)
    
    return pd.DataFrame(calibration_results)