"""
Prompt Management Module
Handles loading and formatting of prompts for sentiment analysis
"""

import os
from typing import Dict, Optional
import pandas as pd


class PromptManager:
    """
    Manages prompt templates and generates formatted prompts for sentiment analysis
    """
    
    def __init__(self, prompt_dir: str = "data/prompts"):
        """
        Initialize the prompt manager
        
        Args:
            prompt_dir: Directory containing prompt template files
        """
        self.prompt_dir = prompt_dir
        self._templates = {}
        
    def load_template(self, template_name: str) -> str:
        """
        Load a prompt template from file
        
        Args:
            template_name: Name of the template file (without .txt extension)
            
        Returns:
            str: The loaded template content
        """
        if template_name in self._templates:
            return self._templates[template_name]
            
        template_path = os.path.join(self.prompt_dir, f"{template_name}.txt")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            self._templates[template_name] = template
            return template
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    def generate_prompt_for_post(
        self, 
        post_content: str, 
        post_id: str, 
        template_name: str = "zero_shot_prompt"
    ) -> str:
        """
        Generate a formatted prompt for a single post
        
        Args:
            post_content: The content of the post to analyze
            post_id: Unique identifier for the post
            template_name: Name of the prompt template to use
            
        Returns:
            str: Formatted prompt ready for LLM input
        """
        template = self.load_template(template_name)
        
        # Replace placeholders in template
        formatted_prompt = template.replace("[Post]", f"{post_id}: {post_content}")
        formatted_prompt = formatted_prompt.replace("{post_id}", post_id)
        formatted_prompt = formatted_prompt.replace("{post_content}", post_content)
        
        return formatted_prompt
    
    def generate_prompts_for_dataframe(
        self, 
        df: pd.DataFrame, 
        post_id_col: str = "PostId", 
        content_col: str = "Body",
        template_name: str = "zero_shot_prompt"
    ) -> Dict[str, str]:
        """
        Generate prompts for all posts in a DataFrame
        
        Args:
            df: DataFrame containing posts
            post_id_col: Column name for post IDs
            content_col: Column name for post content
            template_name: Name of the prompt template to use
            
        Returns:
            Dict[str, str]: Mapping from post_id to formatted prompt
        """
        prompts = {}
        
        for _, row in df.iterrows():
            post_id = str(row[post_id_col])
            content = str(row[content_col])
            
            prompt = self.generate_prompt_for_post(
                post_content=content,
                post_id=post_id,
                template_name=template_name
            )
            prompts[post_id] = prompt
            
        return prompts