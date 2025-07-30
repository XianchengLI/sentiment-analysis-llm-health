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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


    # add confidence

    def run_single_model_experiment_with_confidence(
        self, 
        df: pd.DataFrame,
        model_name: str,
        prompt_template: str = "zero_shot",
        post_id_col: str = "PostId",
        content_col: str = "Body",
        expert_label_col: str = "Expert_Label"
    ) -> Dict:
        """
        Run experiment with a single model including confidence scores
        """
        self.logger.info(f"Running confidence experiment with {model_name} using {prompt_template}")
        
        # Generate prompts with confidence
        prompts = self.prompt_manager.generate_prompts_for_dataframe_with_confidence(
            df=df,
            post_id_col=post_id_col,
            content_col=content_col,
            template_name=prompt_template,
            with_confidence=True
        )
        
        # Get model predictions
        prompt_list = [prompts[str(post_id)] for post_id in df[post_id_col]]
        post_ids = [str(post_id) for post_id in df[post_id_col]]
        
        responses = self.llm_client.batch_generate_labels_with_confidence(
            prompts=prompt_list,
            model=model_name,
            show_progress=True
        )
        
        # Parse responses with confidence
        parsed_results = ResponseParser.parse_batch_responses_with_confidence(responses, post_ids)
        
        # Create results DataFrame
        results_df = pd.DataFrame(parsed_results)
        if not results_df.empty:
            results_df[post_id_col] = results_df['PostId'].astype(df[post_id_col].dtype)
            
            # Merge with original data
            df_with_predictions = df.merge(
                results_df[[post_id_col, 'Sentiment', 'Confidence']], 
                on=post_id_col, 
                how='left'
            )
            
            # Rename columns to include model name
            df_with_predictions = df_with_predictions.rename(columns={
                'Sentiment': f'Predicted_{model_name}',
                'Confidence': f'Confidence_{model_name}'
            })
        else:
            self.logger.error("No valid responses parsed")
            return {}
        
        # Calculate metrics (same as before)
        if expert_label_col in df_with_predictions.columns:
            metrics = self.evaluator.calculate_metrics(
                y_true=df_with_predictions[expert_label_col],
                y_pred=df_with_predictions[f'Predicted_{model_name}']
            )
            
            # Add confidence-specific metrics
            if f'Confidence_{model_name}' in df_with_predictions.columns:
                confidence_scores = df_with_predictions[f'Confidence_{model_name}'].dropna()
                metrics['mean_confidence'] = confidence_scores.mean()
                metrics['std_confidence'] = confidence_scores.std()
                metrics['min_confidence'] = confidence_scores.min()
                metrics['max_confidence'] = confidence_scores.max()
        else:
            metrics = {}
        
        result = {
            'model': model_name,
            'prompt_template': prompt_template,
            'with_confidence': True,
            'metrics': metrics,
            'predictions': df_with_predictions,
            'response_count': len([r for r in responses if r is not None]),
            'total_count': len(responses)
        }
        
        return result

def run_experiment_with_confidence(
    data_path: str,
    models: List[str],
    post_id_col: str = "PostId",
    content_col: str = "Body", 
    expert_label_col: str = "Expert_Label",
    prompt_templates: List[str] = None,
    output_dir: str = "results/confidence_experiments/",
    verbose: bool = True
) -> Dict:
    """
    Run sentiment analysis experiment with confidence scores
    """
    if prompt_templates is None:
        prompt_templates = ["zero_shot", "few_shot", "naive"]
    
    # Initialize experiment
    experiment = SentimentExperiment()
    
    # Load data
    if verbose:
        print(f" Loading data from: {data_path}")
    df = experiment.load_data(data_path)
    if verbose:
        print(f" Loaded {len(df)} posts")
    
    # Run experiments with confidence
    all_results = {}
    
    for model in models:
        for template in prompt_templates:
            experiment_name = f"{model}_{template}_confidence"
            if verbose:
                print(f"\n Running confidence experiment: {experiment_name}")
            
            try:
                result = experiment.run_single_model_experiment_with_confidence(
                    df=df,
                    model_name=model,
                    prompt_template=template,
                    post_id_col=post_id_col,
                    content_col=content_col,
                    expert_label_col=expert_label_col
                )
                all_results[experiment_name] = result
                
                # Save results
                if result and 'predictions' in result:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{experiment_name}_predictions.csv")
                    result['predictions'].to_csv(output_path, index=False)
                    if verbose:
                        print(f" Results saved to: {output_path}")
                    
                    # Print metrics
                    metrics = result.get('metrics', {})
                    accuracy = metrics.get('accuracy', 0)
                    mean_conf = metrics.get('mean_confidence', 0)
                    if verbose:
                        print(f" Accuracy: {accuracy:.3f}, Mean Confidence: {mean_conf:.3f}")
                        
            except Exception as e:
                if verbose:
                    print(f" Error in experiment {experiment_name}: {e}")
                all_results[experiment_name] = {'error': str(e)}
    
    return all_results

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

def predict_sentiment_batch(
    data_path: str,
    models: List[str],
    post_id_col: str = "PostId",
    content_col: str = "Body",
    prompt_template: str = "few_shot_prompt", 
    output_dir: str = "results/predictions/",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Predict sentiment for unlabeled data (no ground truth needed)
    
    Args:
        data_path: Path to the dataset file (CSV or Excel)
        models: List of model names to use (e.g., ["gpt-4o-mini", "llama3.1-70b"])
        post_id_col: Column name for post IDs
        content_col: Column name for post content
        prompt_template: Prompt template to use (default: "few_shot_prompt")
        output_dir: Directory to save predictions
        verbose: Whether to print detailed progress information
        
    Returns:
        pd.DataFrame: Original data with predicted sentiment columns for each model
        
    Example:
        df_results = predict_sentiment_batch(
            data_path="new_posts.csv",
            models=["gpt-4o-mini", "llama3.1-70b"],
            post_id_col="ID",
            content_col="PostText"
        )
    """
    
    # Initialize experiment
    experiment = SentimentExperiment()
    
    # Path adjustments for running from notebooks
    if os.path.exists("../data/prompts"):
        experiment.prompt_manager = PromptManager(prompt_dir="../data/prompts")
        if not output_dir.startswith("../"):
            output_dir = "../" + output_dir
    
    # Load data
    if verbose:
        print(f" Loading data from: {data_path}")
    df = experiment.load_data(data_path)
    if verbose:
        print(f" Loaded {len(df)} posts for prediction")
    
    # Validate required columns
    required_cols = [post_id_col, content_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    if verbose:
        print(f" Using columns: PostID='{post_id_col}', Content='{content_col}'")
        print(f" Predicting sentiment using prompt: {prompt_template}")
    
    # Initialize results DataFrame
    results_df = df.copy()
    
    # Run predictions for each model
    for model in models:
        if verbose:
            print(f"\n Predicting with {model}...")
        
        try:
            # Generate prompts
            prompts = experiment.prompt_manager.generate_prompts_for_dataframe(
                df=df,
                post_id_col=post_id_col,
                content_col=content_col,
                template_name=prompt_template
            )
            
            # Get predictions
            prompt_list = [prompts[str(post_id)] for post_id in df[post_id_col]]
            post_ids = [str(post_id) for post_id in df[post_id_col]]
            
            responses = experiment.llm_client.batch_generate_labels(
                prompts=prompt_list,
                model=model,
                show_progress=verbose
            )
            
            # Parse responses
            parsed_results = ResponseParser.parse_batch_responses(responses, post_ids)
            
            # Add predictions to results
            if parsed_results:
                prediction_df = pd.DataFrame(parsed_results)
                prediction_df[post_id_col] = prediction_df['PostId'].astype(df[post_id_col].dtype)
                
                # Merge predictions
                results_df = results_df.merge(
                    prediction_df[[post_id_col, 'Sentiment']].rename(
                        columns={'Sentiment': f'Predicted_{model}'}
                    ),
                    on=post_id_col,
                    how='left'
                )
                
                # Calculate response rate
                response_count = len([r for r in responses if r is not None])
                response_rate = response_count / len(responses)
                
                if verbose:
                    print(f" {model}: {response_count}/{len(responses)} predictions ({response_rate:.1%} success rate)")
                    
                    # Show prediction distribution
                    pred_col = f'Predicted_{model}'
                    if pred_col in results_df.columns:
                        distribution = results_df[pred_col].value_counts().to_dict()
                        print(f" Distribution: {distribution}")
            else:
                if verbose:
                    print(f"❌ {model}: No valid predictions generated")
                # Add empty column for consistency
                results_df[f'Predicted_{model}'] = None
                
        except Exception as e:
            error_msg = f"❌ Error with {model}: {e}"
            if verbose:
                print(error_msg)
            # Add empty column for failed model
            results_df[f'Predicted_{model}'] = None
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sentiment_predictions_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n Predictions saved to: {output_path}")
        print(f" Results summary:")
        print(f"   - Total posts: {len(results_df)}")
        print(f"   - Models used: {models}")
        
        # Show columns added
        new_cols = [col for col in results_df.columns if col.startswith('Predicted_')]
        print(f"   - New columns: {new_cols}")
    
    return results_df

def predict_sentiment_batch_with_confidence(
    data_path: str,
    models: List[str],
    post_id_col: str = "PostId",
    content_col: str = "Body",
    prompt_template: str = "few_shot", 
    output_dir: str = "results/predictions/",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Predict sentiment with confidence for unlabeled data (no ground truth needed)
    
    Args:
        data_path: Path to the dataset file (CSV or Excel)
        models: List of model names to use (e.g., ["gpt-4o-mini", "llama3.1-70b"])
        post_id_col: Column name for post IDs
        content_col: Column name for post content
        prompt_template: Base template name to use (default: "few_shot")
        output_dir: Directory to save predictions
        verbose: Whether to print detailed progress information
        
    Returns:
        pd.DataFrame: Original data with predicted sentiment and confidence columns for each model
    """
    
    # Initialize experiment
    experiment = SentimentExperiment()
    
    # Path adjustments for running from notebooks
    if os.path.exists("../data/prompts"):
        experiment.prompt_manager = PromptManager(prompt_dir="../data/prompts")
        if not output_dir.startswith("../"):
            output_dir = "../" + output_dir
    
    # Load data
    if verbose:
        print(f" Loading data from: {data_path}")
    df = experiment.load_data(data_path)
    if verbose:
        print(f" Loaded {len(df)} posts for confidence prediction")
    
    # Validate required columns
    required_cols = [post_id_col, content_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    if verbose:
        print(f" Using columns: PostID='{post_id_col}', Content='{content_col}'")
        print(f" Predicting sentiment with confidence using template: {prompt_template}")
    
    # Initialize results DataFrame
    results_df = df.copy()
    
    # Run predictions for each model
    for model in models:
        if verbose:
            print(f"\n Predicting with confidence using {model}...")
        
        try:
            # Use the existing confidence experiment method
            result = experiment.run_single_model_experiment_with_confidence(
                df=df,
                model_name=model,
                prompt_template=prompt_template,
                post_id_col=post_id_col,
                content_col=content_col,
                expert_label_col=None  # No expert labels for pure prediction
            )
            
            if 'predictions' in result:
                pred_df = result['predictions']
                
                # Extract the new columns
                pred_col = f'Predicted_{model}'
                conf_col = f'Confidence_{model}'
                
                if pred_col in pred_df.columns and conf_col in pred_df.columns:
                    # Merge predictions and confidence
                    results_df = results_df.merge(
                        pred_df[[post_id_col, pred_col, conf_col]],
                        on=post_id_col,
                        how='left'
                    )
                    
                    # Calculate success metrics
                    valid_preds = pred_df[pred_col].notna().sum()
                    total_preds = len(pred_df)
                    success_rate = valid_preds / total_preds if total_preds > 0 else 0
                    
                    if verbose:
                        print(f" {model}: {valid_preds}/{total_preds} predictions ({success_rate:.1%} success rate)")
                        
                        # Show confidence statistics
                        if conf_col in results_df.columns:
                            conf_scores = results_df[conf_col].dropna()
                            if len(conf_scores) > 0:
                                print(f" Mean confidence: {conf_scores.mean():.3f}")
                                
                                # Show prediction distribution
                                if pred_col in results_df.columns:
                                    distribution = results_df[pred_col].value_counts().to_dict()
                                    print(f" Distribution: {distribution}")
                else:
                    if verbose:
                        print(f"  {model}: Expected columns not found in results")
            else:
                if verbose:
                    print(f" {model}: No valid predictions generated")
                # Add empty columns for consistency
                results_df[f'Predicted_{model}'] = None
                results_df[f'Confidence_{model}'] = None
                
        except Exception as e:
            error_msg = f" Error with {model}: {e}"
            if verbose:
                print(error_msg)
            # Add empty columns for failed model
            results_df[f'Predicted_{model}'] = None
            results_df[f'Confidence_{model}'] = None
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"confidence_predictions_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n Confidence predictions saved to: {output_path}")
        print(f" Results summary:")
        print(f"   - Total posts: {len(results_df)}")
        print(f"   - Models used: {models}")
        
        # Show columns added
        new_cols = [col for col in results_df.columns if col.startswith(('Predicted_', 'Confidence_'))]
        print(f"   - New columns: {new_cols}")
    
    return results_df

if __name__ == "__main__":
    run_simple_experiment()