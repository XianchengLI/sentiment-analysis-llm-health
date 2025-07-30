"""
Prompt Management Module
Handles loading and formatting of prompts for sentiment analysis
"""

import os
from typing import Dict, Optional
import pandas as pd
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompt templates and generates formatted prompts for sentiment analysis
    """
    
    def __init__(self, prompt_dir: str = "../data/prompts"):
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


    # add confidence

    def _load_confidence_guidelines(self) -> str:
        """Load the unified confidence guidelines"""
        guidelines_path = os.path.join(self.prompt_dir, "confidence_guidelines.txt")
        try:
            with open(guidelines_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Confidence guidelines not found at {guidelines_path}")
            return ""

    def _load_base_template(self, template_name: str) -> str:
        """Load base template (without response format)"""
        # Remove '_base' suffix if present and add it
        base_name = template_name.replace('_prompt', '').replace('_base', '')
        base_path = Path(self.prompt_dir) / "base_templates" / f"{base_name}_base.txt"
     
        try:
            with open(base_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Base template not found at {base_path}")
            # Fallback to original template
            return ""

    def generate_prompt_with_confidence(
        self, 
        template_name: str, 
        post_content: str, 
        post_id: str,
        with_confidence: bool = True
    ) -> str:
        """Generate prompt with optional confidence assessment"""
        
        # Load base template
        base_template = self._load_base_template(template_name)
        
        # Prepare confidence section
        if with_confidence:
            confidence_guidelines = self._load_confidence_guidelines()
            confidence_section = f"\n{confidence_guidelines}\n"
            response_format = f"[Your Response Format]\n{post_id}: <Positive|Neutral|Negative>, <confidence_score>"
        else:
            confidence_section = ""
            response_format = f"[Your Response Format]\n{post_id}: <Positive|Neutral|Negative>"
        
        # Format the complete prompt
        try:
            formatted_prompt = base_template.format(
                confidence_section=confidence_section,
                response_format=response_format,
                post_content=post_content,
                post_id=post_id
            )
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Template formatting error: {e}")
            # Fallback to original method
            return self.generate_prompt_for_post(post_content, post_id, template_name)

    def generate_prompts_for_dataframe_with_confidence(
        self,
        df: pd.DataFrame,
        post_id_col: str = "PostId",
        content_col: str = "Body", 
        template_name: str = "zero_shot",
        with_confidence: bool = True
    ) -> Dict[str, str]:
        """Generate prompts for entire dataframe with confidence option"""
        
        prompts = {}
        for _, row in df.iterrows():
            post_id = str(row[post_id_col])
            content = str(row[content_col])
            
            prompt = self.generate_prompt_with_confidence(
                template_name=template_name,
                post_content=content,
                post_id=post_id,
                with_confidence=with_confidence
            )
            prompts[post_id] = prompt
        
        return prompts