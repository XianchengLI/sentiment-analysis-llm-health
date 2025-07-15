"""
Main Experiment Runner
Orchestrates the sentiment analysis experiments with different LLM models
"""

import os
import sys
import logging
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.llm_client import LLMClient, ResponseParser
from data.prompt_manager import PromptManager
from evaluation.metrics import SentimentEvaluator
from data.sample_generator import create_sample_dataset


class SentimentExperiment:
    """
    Main experiment class for running sentiment analysis with LLMs
    """
    
    def __init__(self):
        """Initialize the experiment"""
        
        # Initialize components
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()
        self.evaluator = SentimentEvaluator()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load dataset for analysis
        """
        if data_path.endswith('.csv'):
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    return pd.read_csv(data_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {data_path} with any of the tried encodings: {encodings}")
        elif data_path.endswith('.xlsx'):
            return pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def run_single_model_experiment(
        self, 
        df: pd.DataFrame,
        model_name: str,
        prompt_template: str = "zero_shot_prompt",
        post_id_col: str = "PostId",
        content_col: str = "Body",
        expert_label_col: str = "Expert_Label"
    ) -> Dict:
        """
        Run experiment with a single model
        
        Args:
            df: Dataset DataFrame
            model_name: Name of the LLM model to use
            prompt_template: Name of prompt template
            post_id_col: Column name for post IDs
            content_col: Column name for post content
            expert_label_col: Column name for expert labels
            
        Returns:
            Dict: Experiment results
        """
        self.logger.info(f"Running experiment with {model_name} using {prompt_template}")
        
        # Generate prompts
        prompts = self.prompt_manager.generate_prompts_for_dataframe(
            df=df,
            post_id_col=post_id_col,
            content_col=content_col,
            template_name=prompt_template
        )
        
        # Get model predictions
        prompt_list = [prompts[str(post_id)] for post_id in df[post_id_col]]
        post_ids = [str(post_id) for post_id in df[post_id_col]]
        
        responses = self.llm_client.batch_generate_labels(
            prompts=prompt_list,
            model=model_name,
            show_progress=True
        )
        
        # Parse responses
        parsed_results = ResponseParser.parse_batch_responses(responses, post_ids)
        
        # Create results DataFrame
        results_df = pd.DataFrame(parsed_results)
        if not results_df.empty:
            results_df[post_id_col] = results_df['PostId'].astype(df[post_id_col].dtype)
            
            # Merge with original data
            df_with_predictions = df.merge(
                results_df[[post_id_col, 'Sentiment']], 
                on=post_id_col, 
                how='left'
            )
            df_with_predictions = df_with_predictions.rename(
                columns={'Sentiment': f'Predicted_{model_name}'}
            )
        else:
            self.logger.error("No valid responses parsed")
            return {}
        
        # Calculate metrics
        if expert_label_col in df_with_predictions.columns:
            metrics = self.evaluator.calculate_metrics(
                y_true=df_with_predictions[expert_label_col],
                y_pred=df_with_predictions[f'Predicted_{model_name}']
            )
        else:
            metrics = {}
            self.logger.warning(f"No expert labels found in column {expert_label_col}")
        
        # Store results
        result = {
            'model': model_name,
            'prompt_template': prompt_template,
            'metrics': metrics,
            'predictions': df_with_predictions,
            'response_count': len([r for r in responses if r is not None]),
            'total_count': len(responses)
        }
        
        return result


def run_simple_experiment():
    """Run a simple experiment for testing"""
    print(" Running simple sentiment analysis experiment...")
    
    # Check if sample data exists
    data_path = "data/sample_data/sample_posts_test.csv"
    if not os.path.exists(data_path):
        print(" Creating sample data...")
        create_sample_dataset(n_samples=20)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f" Loaded {len(df)} sample posts")
    
    # Initialize experiment
    experiment = SentimentExperiment()
    
    # Run experiment with first 5 posts
    test_posts = df.head(5)
    results = []
    
    for _, row in test_posts.iterrows():
        post_id = row['PostId']
        content = row['Body']
        true_label = row['Expert_Label']
        
        # Generate prompt
        prompt = experiment.prompt_manager.generate_prompt_for_post(
            post_content=content,
            post_id=post_id,
            template_name="zero_shot_prompt"
        )
        
        print(f"\n Analyzing: {post_id}")
        print(f"Content: {content[:100]}...")
        print(f"True label: {true_label}")
        
        # Check if API key is available
        if not os.getenv('OPENAI_API_KEY'):
            print(" No OpenAI API key found. Using mock prediction.")
            predicted_label = "Neutral"
        else:
            # Get prediction
            response = experiment.llm_client.generate_sentiment_label(prompt, model="o3-mini")
            parsed = ResponseParser.parse_sentiment_response(response, post_id)
            predicted_label = parsed['Sentiment'] if parsed else 'Unknown'
        
        print(f"Predicted: {predicted_label}")
        print(f"Match: {'Yes' if predicted_label == true_label else 'No'}")
        
        results.append({
            'PostId': post_id,
            'Content': content,
            'True_Label': true_label,
            'Predicted_Label': predicted_label,
            'Match': predicted_label == true_label
        })
    
    # Summary
    matches = sum(1 for r in results if r['Match'])
    accuracy = matches / len(results)
    
    print(f"\n Results Summary:")
    print(f"Total posts: {len(results)}")
    print(f"Correct predictions: {matches}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results
    os.makedirs("results/experiments", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/experiments/simple_experiment_results.csv", index=False)
    print(f" Results saved to: results/experiments/simple_experiment_results.csv")

def run_experiment_with_custom_data(
    data_path: str,
    models: List[str],
    post_id_col: str = "PostId",
    content_col: str = "Body", 
    expert_label_col: str = "Expert_Label",
    prompt_templates: List[str] = None,
    output_dir: str = "results/custom_experiments/",
    verbose: bool = True  # 新增参数
) -> Dict:
    """
    Run sentiment analysis experiment with custom data
    
    Args:
        data_path: Path to the dataset file (CSV or Excel)
        models: List of model names to test (e.g., ["gpt-4o-mini", "o3"])
        post_id_col: Column name for post IDs
        content_col: Column name for post content
        expert_label_col: Column name for expert labels
        prompt_templates: List of prompt templates (default: ["zero_shot_prompt", "few_shot_prompt"])
        output_dir: Directory to save results
        verbose: Whether to print detailed progress information (default: True)
        
    Returns:
        Dict: Comprehensive experiment results
    """
    if prompt_templates is None:
        prompt_templates = ["zero_shot_prompt", "few_shot_prompt"]
    
    # Initialize experiment
    experiment = SentimentExperiment()
    
    # Path adjustments for running from notebooks
    if os.path.exists("../data/prompts"):
        experiment.prompt_manager = PromptManager(prompt_dir="../data/prompts")
        if not output_dir.startswith("../"):
            output_dir = "../" + output_dir
    
    # Load custom data
    if verbose:
        print(f" Loading data from: {data_path}")
    df = experiment.load_data(data_path)
    if verbose:
        print(f" Loaded {len(df)} posts")
    
    # Validate required columns
    required_cols = [post_id_col, content_col, expert_label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    if verbose:
        print(f" Using columns: PostID='{post_id_col}', Content='{content_col}', Label='{expert_label_col}'")
        print(f" Label distribution: {df[expert_label_col].value_counts().to_dict()}")
    
    # Run experiments
    all_results = {}
    
    for model in models:
        for template in prompt_templates:
            experiment_name = f"{model}_{template}"
            if verbose:
                print(f"\n Running experiment: {experiment_name}")
            
            try:
                result = experiment.run_single_model_experiment(
                    df=df,
                    model_name=model,
                    prompt_template=template,
                    post_id_col=post_id_col,
                    content_col=content_col,
                    expert_label_col=expert_label_col
                )
                all_results[experiment_name] = result
                
                # Save individual results
                if result and 'predictions' in result:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{experiment_name}_predictions.csv")
                    result['predictions'].to_csv(output_path, index=False)
                    if verbose:
                        print(f" Results saved to: {output_path}")
                    
                    # Print quick summary
                    metrics = result.get('metrics', {})
                    accuracy = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1_macro', 0)
                    response_rate = metrics.get('response_rate', 0)
                    if verbose:
                        print(f" Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Response Rate: {response_rate:.3f}")
                    
            except Exception as e:
                error_msg = f" Error in experiment {experiment_name}: {e}"
                if verbose:
                    print(error_msg)
                all_results[experiment_name] = {'error': str(e)}
    
    # Generate comparison summary
    if verbose:
        print(f"\n Generating comparison summary...")
    
    comparison_data = []
    for experiment_name, result in all_results.items():
        if 'error' in result:
            continue
            
        metrics = result.get('metrics', {})
        row = {
            'Experiment': experiment_name,
            'Model': result.get('model', ''),
            'Template': result.get('prompt_template', ''),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision_Macro': metrics.get('precision_macro', 0),
            'Recall_Macro': metrics.get('recall_macro', 0),
            'F1_Macro': metrics.get('f1_macro', 0),
            'Response_Rate': result.get('response_count', 0) / max(result.get('total_count', 1), 1)
        }
        comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(output_dir, "comparison_summary.csv")
        comparison_df.to_csv(comparison_path, index=False)
        if verbose:
            print(f" Comparison summary saved to: {comparison_path}")
            
            # Print summary table
            print(f"\n Results Summary:")
            print(comparison_df[['Experiment', 'Accuracy', 'F1_Macro', 'Response_Rate']].round(3).to_string(index=False))
    
    if verbose:
        print(f"\n Experiment completed! Results saved to: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    run_simple_experiment()