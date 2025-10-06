import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, Union
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

from ..evaluation.confidence_calibration import (
    ConfidenceCalibrationAnalyzer, 
    ConfidenceVisualizer,
    analyze_confidence_calibration
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ConfidenceAnalysisRunner:
    """
    Runner for confidence calibration analysis with publication-ready visualizations
    """
    
    def __init__(
        self, 
        data_path: str = "data/sample_data/example_data_no_text.csv",
        sentiment_mapping: Dict[Union[int, str], str] = None,
        true_label_col: str = "Sentiment_Majority"
    ):
        self.data_path = Path(data_path)
        self.sentiment_mapping = sentiment_mapping or {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        self.true_label_col = true_label_col
        
        # Set up plotting style for publication
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
    
    def detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that contain numeric sentiment labels"""
        numeric_cols = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                unique_vals = set(df[col].dropna().unique())
                mapping_keys = set(self.sentiment_mapping.keys())
                
                if unique_vals.issubset(mapping_keys):
                    numeric_cols.append(col)
                    logger.info(f"Detected numeric sentiment column: {col} with values {sorted(unique_vals)}")
        
        return numeric_cols
    
    def apply_sentiment_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sentiment mapping to detected numeric columns"""
        df_copy = df.copy()
        numeric_cols = self.detect_numeric_columns(df_copy)
        
        for col in numeric_cols:
            original_col = f"{col}_original"
            df_copy[original_col] = df_copy[col]
            df_copy[col] = df_copy[col].map(self.sentiment_mapping)
            logger.info(f"Mapped column {col}: {dict(zip(df_copy[original_col].unique(), df_copy[col].unique()))}")
        
        return df_copy
    
    def load_and_transform_data(self) -> pd.DataFrame:
        """Load wide-format data and transform to long format for analysis"""
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} posts from {self.data_path}")
        
        df = self.apply_sentiment_mapping(df)
        
        if self.true_label_col not in df.columns:
            available_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]
            raise ValueError(f"True label column '{self.true_label_col}' not found. Available: {available_cols}")
        
        # Handle true sentiment column
        if df[self.true_label_col].dtype in ['int64', 'float64']:
            df['True_Sentiment'] = df[self.true_label_col].map(self.sentiment_mapping)
            if df['True_Sentiment'].isna().any():
                logger.warning(f"Some values in {self.true_label_col} couldn't be mapped")
        else:
            df['True_Sentiment'] = df[self.true_label_col]
        
        logger.info(f"True sentiment distribution: {df['True_Sentiment'].value_counts().to_dict()}")
        
        # Extract model configurations
        model_configs = self._extract_model_configs(df.columns)
        logger.info(f"Found {len(model_configs)} model configurations")
        
        # Transform to long format
        long_data = []
        for model, prompt_type in model_configs:
            pred_col = f'Predicted_{model}_{prompt_type}'
            conf_col = f'Confidence_{model}_{prompt_type}'
            
            if pred_col not in df.columns or conf_col not in df.columns:
                logger.warning(f"Missing columns for {model}_{prompt_type}")
                continue
                
            valid_mask = df[pred_col].notna() & df[conf_col].notna() & df['True_Sentiment'].notna()
            valid_data = df[valid_mask]
            
            if len(valid_data) == 0:
                logger.warning(f"No valid data for {model}_{prompt_type}")
                continue
            
            for _, row in valid_data.iterrows():
                long_data.append({
                    'PostId': row['PostId'],
                    'Model': model,
                    'Prompt_Type': prompt_type,
                    'Model_Config': f"{model}_{prompt_type}",
                    'Predicted_Sentiment': row[pred_col],
                    'Confidence': row[conf_col],
                    'True_Sentiment': row['True_Sentiment']
                })
        
        result_df = pd.DataFrame(long_data)
        logger.info(f"Transformed to {len(result_df)} model-post combinations")
        logger.info(f"Model configurations: {sorted(result_df['Model_Config'].unique())}")
        
        return result_df
    
    def _extract_model_configs(self, columns: List[str]) -> List[Tuple[str, str]]:
        """Extract model and prompt type combinations from column names"""
        configs = []
        
        for col in columns:
            if col.startswith('Predicted_'):
                parts = col.replace('Predicted_', '').split('_')
                
                if len(parts) >= 2:
                    prompt_types = ['zero_shot', 'few_shot', 'naive']
                    
                    for i in range(len(parts)-1, 0, -1):
                        potential_prompt = '_'.join(parts[i:])
                        if potential_prompt in prompt_types:
                            model = '_'.join(parts[:i])
                            prompt_type = potential_prompt
                            configs.append((model, prompt_type))
                            break
                    else:
                        model = '_'.join(parts[:-1])
                        prompt_type = parts[-1]
                        configs.append((model, prompt_type))
        
        unique_configs = list(set(configs))
        logger.info(f"Extracted model configurations: {unique_configs}")
        return unique_configs
    
    def prepare_detailed_results(self, df_long: pd.DataFrame) -> Dict:
        """Prepare detailed results dictionary for visualization"""
        detailed_results = {}
        
        # Add accuracy column
        df_long['Is_Correct'] = (df_long['Predicted_Sentiment'] == df_long['True_Sentiment']).astype(int)
        
        # Group by model configuration
        for model_config in df_long['Model_Config'].unique():
            model_data = df_long[df_long['Model_Config'] == model_config].copy()
            
            # Create confidence column name in expected format
            conf_col_name = f"Confidence_{model_config}"
            model_data[conf_col_name] = model_data['Confidence']
            
            detailed_results[model_config] = model_data
        
        logger.info(f"Prepared detailed results for {len(detailed_results)} model configurations")
        return detailed_results
    
    def create_calibration_plot(self, confidence_scores, accuracy_labels, experiment_name, ax1, ax2):
        """
        Create calibration plot with histogram and calibration curve
        
        Args:
            confidence_scores: Array of confidence scores (0-1)
            accuracy_labels: Array of binary accuracy labels (0 or 1)
            experiment_name: Name for the plot title
            ax1: Axis for histogram (left)
            ax2: Axis for calibration curve (right)
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(confidence_scores) | np.isnan(accuracy_labels))
        conf_clean = confidence_scores[valid_mask]
        acc_clean = accuracy_labels[valid_mask]
        
        if len(conf_clean) == 0:
            ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
            return
        
        # Left plot: Confidence histogram
        ax1.hist(conf_clean, bins=15, alpha=0.7, color='mediumpurple', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Confidence score')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, 1)
        ax1.set_title(experiment_name, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Calibration curve
        # Create confidence bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        
        for i in range(n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]
            
            # Find samples in this bin
            if i == n_bins - 1:  # Last bin includes the upper boundary
                in_bin = (conf_clean >= bin_start) & (conf_clean <= bin_end)
            else:
                in_bin = (conf_clean >= bin_start) & (conf_clean < bin_end)
            
            if np.sum(in_bin) > 0:
                bin_center = (bin_start + bin_end) / 2
                bin_accuracy = np.mean(acc_clean[in_bin])
                bin_centers.append(bin_center)
                bin_accuracies.append(bin_accuracy)
        
        # Plot calibration curve
        if bin_centers:
            ax2.plot(bin_centers, bin_accuracies, 'o-', color='darkblue', linewidth=2, markersize=6)
            ax2.step(bin_centers, bin_accuracies, where='mid', color='darkblue', alpha=0.7, linewidth=2)
        
        # Perfect calibration line (diagonal)
        ax2.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, linewidth=1, label='Perfect calibration')
        
        ax2.set_xlabel('Confidence (C)')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
    
    def calculate_ece(self, confidence_scores, accuracy_labels):
        """Calculate Expected Calibration Error"""
        valid_mask = ~(np.isnan(confidence_scores) | np.isnan(accuracy_labels))
        conf_clean = confidence_scores[valid_mask]
        acc_clean = accuracy_labels[valid_mask]
        
        if len(conf_clean) == 0:
            return np.nan
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        total_samples = len(conf_clean)
        
        for i in range(n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]
            
            if i == n_bins - 1:
                in_bin = (conf_clean >= bin_start) & (conf_clean <= bin_end)
            else:
                in_bin = (conf_clean >= bin_start) & (conf_clean < bin_end)
            
            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(conf_clean[in_bin])
                bin_accuracy = np.mean(acc_clean[in_bin])
                bin_weight = np.sum(in_bin) / total_samples
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def run_calibration_visualization(
        self,
        models_to_test: List[str] = None,
        prompt_templates: List[str] = None,
        output_dir: str = "results/confidence_analysis"
    ) -> Dict:
        """
        Run complete calibration visualization analysis
        
        Args:
            models_to_test: List of models to analyze (e.g., ["o3", "o3-mini"])
            prompt_templates: List of prompt types (e.g., ["zero_shot", "few_shot", "naive"])
            output_dir: Directory to save results
            
        Returns:
            Dict: Analysis results including plots and metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        if models_to_test is None:
            models_to_test = ["o3", "o3-mini"]
        if prompt_templates is None:
            prompt_templates = ["zero_shot", "few_shot", "naive"]
        
        # Load and transform data
        logger.info("Loading and transforming data...")
        df_long = self.load_and_transform_data()
        
        # Prepare detailed results
        detailed_results = self.prepare_detailed_results(df_long)
        
        # Generate experiment combinations
        selected_experiments = []
        for model in models_to_test:
            for prompt in prompt_templates:
                exp_key = f"{model}_{prompt}"
                display_name = f"{model} + {prompt.replace('_', '-')}"
                selected_experiments.append((exp_key, display_name))
        
        print(f" Total experiments to visualize: {len(selected_experiments)}")
        for exp_key, display_name in selected_experiments:
            print(f"   - {display_name}")
        
        # Create visualization
        print(f"\n Creating calibration plots for all {len(selected_experiments)} experiments...")
        
        n_experiments = len(selected_experiments)
        n_cols = 2  # histogram + calibration curve
        n_rows = n_experiments
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_count = 0
        successful_plots = []
        calibration_summary = []
        
        for i, (exp_key, display_name) in enumerate(selected_experiments):
            if exp_key in detailed_results:
                work_df = detailed_results[exp_key]
                
                # Find confidence column
                conf_col = None
                for col in work_df.columns:
                    if col.startswith('Confidence_') and exp_key in col:
                        conf_col = col
                        break
                
                if conf_col is not None:
                    confidence_scores = work_df[conf_col].values
                    accuracy_labels = work_df['Is_Correct'].values
                    
                    # Create calibration plot
                    self.create_calibration_plot(
                        confidence_scores, 
                        accuracy_labels, 
                        display_name,
                        axes[i, 0], 
                        axes[i, 1]
                    )
                    
                    successful_plots.append(exp_key)
                    plot_count += 1
                    
                    # Calculate metrics
                    if len(confidence_scores) > 0 and not np.all(np.isnan(confidence_scores)):
                        ece = self.calculate_ece(confidence_scores, accuracy_labels)
                        accuracy = np.nanmean(accuracy_labels)
                        mean_confidence = np.nanmean(confidence_scores)
                        
                        calibration_summary.append({
                            'Experiment': exp_key,
                            'Display_Name': display_name,
                            'ECE': ece,
                            'Samples': len(confidence_scores),
                            'Accuracy': accuracy,
                            'Mean_Confidence': mean_confidence
                        })
                else:
                    # No confidence data
                    axes[i, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i, 1].transAxes)
                    axes[i, 0].set_title(display_name, fontsize=12, fontweight='bold')
            else:
                # Missing experiment
                axes[i, 0].text(0.5, 0.5, 'Missing', ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 1].text(0.5, 0.5, 'Missing', ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 0].set_title(display_name, fontsize=12, fontweight='bold')
        
        print(f" Successfully plotted: {plot_count}/{len(selected_experiments)} experiments")
        
        # Save plot
        plt.tight_layout()
        plot_file = output_path / 'calibration_plots.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_file}")        
        
        # Save configuration
        config = {
            'data_path': str(self.data_path),
            'sentiment_mapping': self.sentiment_mapping,
            'true_label_col': self.true_label_col,
            'models_to_test': models_to_test,
            'prompt_templates': prompt_templates
        }
        
        with open(output_path / "analysis_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        results = {
            'figure': fig,
            'calibration_summary': calibration_summary,
            'successful_plots': successful_plots,
            'config': config,
            'detailed_results': detailed_results
        }
        
        print(f" Results saved to: {output_dir}")
        
        return results


def parse_sentiment_mapping(mapping_str: str) -> Dict[Union[int, str], str]:
    """Parse sentiment mapping from command line string"""
    try:
        mapping = json.loads(mapping_str)
        parsed_mapping = {}
        for k, v in mapping.items():
            try:
                parsed_mapping[int(k)] = v
            except ValueError:
                parsed_mapping[k] = v
        return parsed_mapping
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format for sentiment mapping: {mapping_str}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Run confidence calibration visualization analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default models (o3, o3-mini)
  python -m src.scripts.confidence_analysis
  
  # Custom models and prompts
  python -m src.scripts.confidence_analysis --models o3 o3-mini gpt-4.1 --prompts zero_shot few_shot
  
  # Custom sentiment mapping
  python -m src.scripts.confidence_analysis --sentiment-mapping '{"0": "Negative", "1": "Neutral", "2": "Positive"}'
        """
    )
    
    parser.add_argument('--data-path', 
                       default='data/sample_data/example_data_no_text.csv',
                       help='Path to confidence data CSV')
    parser.add_argument('--models', nargs='+',
                       default=['o3', 'o3-mini'],
                       help='Models to analyze (e.g., o3 o3-mini gpt-4.1)')
    parser.add_argument('--prompts', nargs='+',
                       default=['zero_shot', 'few_shot', 'naive'],
                       help='Prompt types to analyze')
    parser.add_argument('--sentiment-mapping',
                       help='JSON string mapping numeric labels to sentiment strings')
    parser.add_argument('--true-label-col',
                       default='Sentiment_Majority',
                       help='Column name containing true sentiment labels')
    parser.add_argument('--output-dir', 
                       default='results/confidence_analysis', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse sentiment mapping
    sentiment_mapping = None
    if args.sentiment_mapping:
        try:
            sentiment_mapping = parse_sentiment_mapping(args.sentiment_mapping)
            logger.info(f"Using custom sentiment mapping: {sentiment_mapping}")
        except ValueError as e:
            logger.error(f"Error parsing sentiment mapping: {e}")
            return 1
    
    # Run analysis
    runner = ConfidenceAnalysisRunner(
        data_path=args.data_path,
        sentiment_mapping=sentiment_mapping,
        true_label_col=args.true_label_col
    )
    
    try:
        results = runner.run_calibration_visualization(
            models_to_test=args.models,
            prompt_templates=args.prompts,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*50)
        print("CONFIDENCE CALIBRATION VISUALIZATION COMPLETED")
        print("="*50)
        print(f"Results saved to: {args.output_dir}")
        print(f"Models analyzed: {args.models}")
        print(f"Prompt templates: {args.prompts}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())