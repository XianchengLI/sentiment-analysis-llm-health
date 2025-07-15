"""
Sample Data Generator - Updated to match actual research examples
Creates demonstration datasets based on the exact few-shot examples from the research
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import random


class HealthCommunitySampleGenerator:
    """
    Generates sample data based on actual research examples
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the sample generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def _get_actual_research_examples(self) -> List[Dict]:
        """
        Get the actual 11 examples from the research few-shot prompt
        
        Returns:
            List[Dict]: Actual examples with posts, labels, and rule categories
        """
        return [
            {
                'post': "I thought I'd need help last night, but I managed to calm my breathing down.",
                'label': 'Positive',
                'rule': 'Improvement and Self-management',
                'rule_number': 1
            },
            {
                'post': "There was a fuss about the drug about ten years ago and I am not sure how widely it is now used but I had it a few times, I think it was before I started taking Prednisolone more regularly. Sorry but as it was so long ago I cannot remember the side effects but I know there was something that went wrong.",
                'label': 'Neutral',
                'rule': 'Uncertainty',
                'rule_number': 2
            },
            {
                'post': "It does not cause Asthma, but makes the existing symptoms worse, so anything we can do to reduce stress, anxiety, depression is a good thing. Unfortunately long-term health conditions such as Asthma do tend to come with anxiety.",
                'label': 'Neutral',
                'rule': 'Objective Info vs. Personal Experience',
                'rule_number': 3
            },
            {
                'post': "My main concern was (still is) that the steroid component of Fostair is Beclometasone. I've been on that steroid before and my asthma was never properly controlled while I was on it. The fluticasone has really been so much more effective.",
                'label': 'Positive',
                'rule': 'Polarized Sentiment from Emphasis',
                'rule_number': 4
            },
            {
                'post': "I was just going to say you can get them on Amazon. Www.powerbreathe.com has them too but it looks like they don't have the flutter. The flutter and the acapella are to help move mucus and are of a great help to people with Bronchiectasis. You can also find breathing exercises online on YouTube for the same purpose but without buying any device. Have a look on YouTube.",
                'label': 'Positive',
                'rule': 'Helpful Advice or Resources',
                'rule_number': 5
            },
            {
                'post': "I do not want to make you all paranoid and suspicious but feel the need to warn everyone to be aware. While things are still new I think that we all need to be careful who we give our contact details to. This is a public forum and anyone can post. Last year we had some problems with people who posted many things that later proved to be untrue and it caused lots of bad feeling and upset.",
                'label': 'Negative',
                'rule': 'Helpful Advice or Resources (Warning without solution)',
                'rule_number': 5
            },
            {
                'post': "The medication isn't working as expected, and I feel worse than before! I can't do anything without feeling breathless, and I feel like my life is on hold.",
                'label': 'Negative',
                'rule': 'Health Struggles, Pain, and Emotional Hardship',
                'rule_number': 8
            },
            {
                'post': "I'm still awake thanks to lungs throwing a major strop and landing me in A&E earlier this evening. Hope you are all managing to sleep well",
                'label': 'Positive',
                'rule': 'Tone Sensitivity',
                'rule_number': 6
            },
            {
                'post': "I've had a tough night with my asthma, can't seem to catch a break. I'll let you know if it improves. Take care.",
                'label': 'Negative',
                'rule': 'Tone Sensitivity (Polite closing only)',
                'rule_number': 6
            },
            {
                'post': "Wish I could sleep, but my lungs have other ideas again!",
                'label': 'Negative',
                'rule': 'Punctuation Sensitivity (Exclamation)',
                'rule_number': 7
            },
            {
                'post': "Not sure if this helps, but I used a vaporizer and felt a bit better after. Hang in there!",
                'label': 'Positive',
                'rule': 'Tone outweighs unclear content',
                'rule_number': 9
            }
        ]
    
    def create_sample_posts(self, n_samples: int = 100, use_only_research_examples: bool = False) -> pd.DataFrame:
        """
        Create a sample dataset based on research examples
        
        Args:
            n_samples: Number of sample posts to generate
            use_only_research_examples: If True, only use the 11 research examples
            
        Returns:
            pd.DataFrame: Sample dataset with PostId, Body, and expert annotations
        """
        
        # Get the actual research examples
        research_examples = self._get_actual_research_examples()
        
        sample_posts = []
        
        # Always include all research examples first
        for i, example in enumerate(research_examples):
            sample_posts.append({
                'PostId': f'RESEARCH_{i+1:02d}',
                'Body': example['post'],
                'Expert_Label': example['label'],
                'Rule_Category': example['rule'],
                'Rule_Number': example['rule_number'],
                'Source': 'Research_Examples'
            })
        
        # If we need more samples and not restricted to research examples only
        if n_samples > len(research_examples) and not use_only_research_examples:
            additional_needed = n_samples - len(research_examples)
            
            # Generate variations based on the research examples
            additional_posts = self._generate_variations_from_research(
                research_examples, 
                additional_needed
            )
            sample_posts.extend(additional_posts)
        
        # If requested samples is less than research examples, truncate
        if n_samples < len(research_examples):
            sample_posts = sample_posts[:n_samples]
        
        # Create DataFrame
        df = pd.DataFrame(sample_posts)
        
        # Add some realistic annotation variability for demonstration
        if not use_only_research_examples and len(df) > len(research_examples):
            df = self._add_annotation_variability(df)
        
        return df
    
    def _generate_variations_from_research(self, research_examples: List[Dict], n_additional: int) -> List[Dict]:
        """
        Generate variations based on the research examples patterns
        
        Args:
            research_examples: Original research examples
            n_additional: Number of additional examples needed
            
        Returns:
            List[Dict]: Additional example posts
        """
        variations = []
        
        # Template patterns based on research examples
        positive_patterns = [
            "Finally managed to control my symptoms with the new treatment approach.",
            "Here's a helpful resource I found that might work for others: breathing techniques really helped me.",
            "Had a difficult episode last night but managed to get through it. Sending support to everyone here.",
            "The new medication has been much more effective than my previous treatment.",
        ]
        
        negative_patterns = [
            "Having another difficult night with my breathing. This is getting exhausting.",
            "The new treatment isn't working as well as I hoped. Feeling quite frustrated.",
            "Can't seem to find relief with any of the treatments I've tried so far!",
        ]
        
        neutral_patterns = [
            "Has anyone had experience with the newer inhaler devices? Wondering about effectiveness.",
            "The research on this treatment shows mixed results from what I've read.",
            "There are several options available but each has different considerations to think about.",
        ]
        
        patterns = {
            'Positive': positive_patterns,
            'Negative': negative_patterns,
            'Neutral': neutral_patterns
        }
        
        for i in range(n_additional):
            # Choose label proportionally to research examples
            labels = [ex['label'] for ex in research_examples]
            label_counts = {l: labels.count(l) for l in set(labels)}
            
            # Weight selection based on original distribution
            label = random.choices(
                list(label_counts.keys()), 
                weights=list(label_counts.values())
            )[0]
            
            # Select pattern and create variation
            pattern = random.choice(patterns[label])
            
            variations.append({
                'PostId': f'VAR_{i+1:03d}',
                'Body': pattern,
                'Expert_Label': label,
                'Rule_Category': 'Generated_Variation',
                'Rule_Number': 0,
                'Source': 'Generated'
            })
        
        return variations
    
    def _add_annotation_variability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simulated annotation variability to reflect real expert disagreement
        Only applied to generated variations, not research examples
        
        Args:
            df: Original DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional annotator columns
        """
        # Simulate 5 expert annotators with some disagreement
        annotator_columns = {}
        
        for annotator_id in range(1, 6):
            annotator_labels = []
            
            for _, row in df.iterrows():
                original_label = row['Expert_Label']
                
                # Research examples have perfect agreement
                if row['Source'] == 'Research_Examples':
                    label = original_label
                else:
                    # Generated examples have some disagreement (15% chance)
                    if random.random() < 0.15:
                        other_labels = [l for l in ['Positive', 'Negative', 'Neutral'] if l != original_label]
                        label = random.choice(other_labels)
                    else:
                        label = original_label
                
                annotator_labels.append(label)
            
            annotator_columns[f'Annotator_{annotator_id}'] = annotator_labels
        
        # Add annotator columns to DataFrame
        for col, labels in annotator_columns.items():
            df[col] = labels
        
        # Calculate majority label
        annotator_cols = [f'Annotator_{i}' for i in range(1, 6)]
        df['Majority_Label'] = df[annotator_cols].mode(axis=1)[0]
        
        return df
    
    def create_research_only_dataset(self) -> pd.DataFrame:
        """
        Create dataset with only the 11 research examples
        
        Returns:
            pd.DataFrame: Dataset with only research examples
        """
        return self.create_sample_posts(n_samples=11, use_only_research_examples=True)
    
    def get_examples_by_rule(self, rule_number: int) -> pd.DataFrame:
        """
        Get examples that demonstrate a specific rule
        
        Args:
            rule_number: Rule number (1-9)
            
        Returns:
            pd.DataFrame: Examples for the specified rule
        """
        df = self.create_research_only_dataset()
        return df[df['Rule_Number'] == rule_number]


def create_sample_dataset(
    n_samples: int = 200, 
    output_dir: str = "data/sample_data/",
    include_research_only: bool = True
) -> None:
    """
    Create and save sample datasets for demonstration
    
    Args:
        n_samples: Number of samples to generate (minimum 11 for research examples)
        output_dir: Directory to save the datasets
        include_research_only: Whether to create a research-examples-only dataset
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    generator = HealthCommunitySampleGenerator()
    
    # Ensure we have at least the 11 research examples
    n_samples = max(n_samples, 11)
    df_full = generator.create_sample_posts(n_samples)
    
    # Create research-only dataset
    if include_research_only:
        df_research = generator.create_research_only_dataset()
        df_research.to_csv(os.path.join(output_dir, "research_examples_only.csv"), index=False)
        print(f"Created research-only dataset with {len(df_research)} examples")
    
    # Save full dataset
    df_full.to_csv(os.path.join(output_dir, "sample_posts_full.csv"), index=False)
    
    # Create train/test splits
    # Use research examples in test set for validation
    research_indices = df_full[df_full['Source'] == 'Research_Examples'].index
    other_indices = df_full[df_full['Source'] != 'Research_Examples'].index
    
    # Test set: all research examples + some generated ones
    test_size = min(50, len(df_full) // 4)
    if len(other_indices) > 0:
        additional_test_indices = np.random.choice(
            other_indices, 
            size=min(test_size - len(research_indices), len(other_indices)), 
            replace=False
        )
        test_indices = list(research_indices) + list(additional_test_indices)
    else:
        test_indices = list(research_indices)
    
    df_test = df_full.loc[test_indices]
    df_train = df_full[~df_full.index.isin(test_indices)]
    
    # Save splits
    df_train.to_csv(os.path.join(output_dir, "sample_posts_train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "sample_posts_test.csv"), index=False)
    
    print(f"Created sample datasets:")
    print(f"  Full dataset: {len(df_full)} posts")
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"  Research examples in test: {len(research_indices)}")
    
    # Show label distribution
    print(f"Label distribution: {df_full['Expert_Label'].value_counts().to_dict()}")
    
    # Show rule distribution for research examples
    research_rules = df_full[df_full['Source'] == 'Research_Examples']['Rule_Category'].value_counts()
    print(f"Research examples by rule: {research_rules.to_dict()}")


if __name__ == "__main__":
    create_sample_dataset()