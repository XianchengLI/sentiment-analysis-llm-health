"""
LLM Client Module for Sentiment Analysis
Handles API calls to different LLM providers with retry mechanisms
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt
import tiktoken

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class LLMClient:
    """
    A unified client for interacting with different LLM APIs
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client with support for multiple providers
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
        """
        self.openai_client = OpenAI(api_key=api_key)
        
        # Additional clients for other providers (lazy initialization)
        self.llama_client = None
        self.deepseek_client = None
        
    def _get_client_and_model(self, model: str):
        """
        Determine which client to use based on model name
        
        Args:
            model: Model name
            
        Returns:
            tuple: (client, actual_model_name)
        """
        if model.startswith("llama"):
            if not self.llama_client:
                llama_key = os.getenv('LLAMA_API_KEY')
                if not llama_key:
                    raise ValueError("LLAMA_API_KEY not found in environment variables")
                self.llama_client = OpenAI(
                    api_key=llama_key,
                    base_url="https://api.llmapi.com"
                )
            return self.llama_client, model
            
        elif model.startswith("deepseek"):
            if not self.deepseek_client:
                ds_key = os.getenv('DEEPSEEK_API_KEY')
                if not ds_key:
                    raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
                self.deepseek_client = OpenAI(
                    api_key=ds_key,
                    base_url="https://api.deepseek.com"
                )
            return self.deepseek_client, model
            
        else:
            # OpenAI models (gpt-*, o3-*, etc.)
            return self.openai_client, model
        
    @retry(
        retry=retry_if_exception_type((APIStatusError, APITimeoutError, APIConnectionError, RateLimitError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def _chat_completion_with_backoff(self, client, **kwargs) -> Dict:
        """
        Make a chat completion request with exponential backoff retry
        
        Args:
            client: The specific client to use (OpenAI, LLaMA, DeepSeek)
            **kwargs: Arguments to pass to the chat completion API
            
        Returns:
            Dict: Response from the API
        """
        return client.chat.completions.create(**kwargs)
    
    def generate_sentiment_label(
        self, 
        prompt: str, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 50
    ) -> Optional[str]:
        """
        Generate sentiment label for a given prompt with multi-provider support
        
        Args:
            prompt: The formatted prompt containing the post and instructions
            model: The model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            str: The generated response or None if failed
        """
        try:
            # Get appropriate client and model
            client, actual_model = self._get_client_and_model(model)
            
            # Prepare API parameters
            if model.startswith("gpt"):
                api_params = {
                    "model": actual_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature":temperature
                }

            # Only add temperature for models that support it
            else:
                api_params = {
                "model": actual_model,
                "messages": [{"role": "user", "content": prompt}]
            }

            if model.startswith(("gpt", "o3")):
                response = self._chat_completion_with_backoff(client, **api_params)
            else:
                response = client.chat.completions.create(**api_params)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating sentiment label with {model}: {e}")
            return None
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """
        Count tokens in text for a given model
        
        Args:
            text: Input text
            model: Model name for tokenization
            
        Returns:
            int: Number of tokens
        """
        try:
            # Only use tiktoken for OpenAI models
            if model.startswith(("gpt", "o3")):
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback for newer models
                    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
                return len(encoding.encode(text))
            else:
                # For non-OpenAI models, use a rough estimation
                # Approximate: 1 token ≈ 4 characters for English
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Could not count tokens for {model}: {e}")
            return len(text) // 4  # Fallback estimation
    
    def batch_generate_labels(
        self, 
        prompts: List[str], 
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 50,
        show_progress: bool = True
    ) -> List[Optional[str]]:
        """
        Generate sentiment labels for a batch of prompts
        
        Args:
            prompts: List of formatted prompts
            model: The model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            show_progress: Whether to show progress bar
            
        Returns:
            List[Optional[str]]: List of generated responses
        """
        responses = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                prompts = tqdm(prompts, desc=f"Processing posts with {model}")
            except ImportError:
                logger.warning("tqdm not available, showing progress without bar")
        
        for prompt in prompts:
            response = self.generate_sentiment_label(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response)
            
        return responses
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models based on configured API keys
        
        Returns:
            List[str]: List of available model names
        """
        available_models = []
        
        # Check OpenAI models
        if os.getenv('OPENAI_API_KEY'):
            available_models.extend([
                "gpt-4.1", 
                "gpt-4.1-mini",
                "o3", 
                "o3-mini"
            ])
        
        # Check LLaMA models
        if os.getenv('LLAMA_API_KEY'):
            available_models.extend([
                "llama3.1-70b", 
                "llama3.1-405b"
            ])
        
        # Check DeepSeek models
        if os.getenv('DEEPSEEK_API_KEY'):
            available_models.extend([
                "deepseek-chat", 
                "deepseek-coder"
            ])
        
        return available_models

    # add confidence

    def generate_sentiment_with_confidence(
        self, 
        prompt: str, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 100  
    ) -> Optional[str]:
        """
        Generate sentiment label with confidence score
        
        Args:
            prompt: The formatted prompt containing confidence requirements
            model: The model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response (increased for confidence)
            
        Returns:
            str: The generated response with sentiment and confidence
        """
        return self.generate_sentiment_label(
            prompt=prompt,
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens
        )

    def batch_generate_labels_with_confidence(
        self, 
        prompts: List[str], 
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 100,
        show_progress: bool = True
    ) -> List[Optional[str]]:
        """
        Generate sentiment labels with confidence for a batch of prompts
        """
        return self.batch_generate_labels(
            prompts=prompts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            show_progress=show_progress
        )

  
class ResponseParser:
    """
    Parser for LLM responses to extract sentiment labels
    """
    
    @staticmethod
    def parse_sentiment_response(response: str, post_id: str) -> Optional[Dict[str, str]]:
        """
        Parse LLM response to extract post ID and sentiment
        
        Args:
            response: Raw response from LLM
            post_id: Expected post ID
            
        Returns:
            Dict[str, str]: Parsed result with PostId and Sentiment
        """
        if not response:
            return None
            
        lines = response.splitlines()
        for line in lines:
            if ":" in line:
                parsed_id, sentiment = line.split(":", 1)
                return {
                    "PostId": parsed_id.strip(),
                    "Sentiment": sentiment.strip()
                }
        
        # Fallback: try to extract sentiment without format
        sentiment_keywords = ["Positive", "Negative", "Neutral"]
        for keyword in sentiment_keywords:
            if keyword.lower() in response.lower():
                return {
                    "PostId": post_id,
                    "Sentiment": keyword
                }
        
        logger.warning(f"Could not parse response: {response}")
        return None
    
    @staticmethod
    def parse_batch_responses(
        responses: List[str], 
        post_ids: List[str]
    ) -> List[Dict[str, str]]:
        """
        Parse a batch of LLM responses
        
        Args:
            responses: List of raw responses from LLM
            post_ids: List of corresponding post IDs
            
        Returns:
            List[Dict[str, str]]: List of parsed results
        """
        parsed_results = []
        
        for response, post_id in zip(responses, post_ids):
            parsed = ResponseParser.parse_sentiment_response(response, post_id)
            if parsed:
                parsed_results.append(parsed)
        
        return parsed_results

    # add confidence

    @staticmethod
    def parse_sentiment_with_confidence_response(response: str, post_id: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Parse LLM response to extract post ID, sentiment, and confidence
        
        Args:
            response: Raw response from LLM (format: "POST_ID: Sentiment, Confidence")
            post_id: Expected post ID
            
        Returns:
            Dict with PostId, Sentiment, and Confidence
        """
        if not response:
            return None
            
        lines = response.splitlines()
        for line in lines:
            if ":" in line and "," in line:
                try:
                    # Parse format: "POST_001: Positive, 0.85"
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        parsed_id = parts[0].strip()
                        sentiment_confidence = parts[1].strip()
                        
                        # Split sentiment and confidence
                        sentiment_parts = sentiment_confidence.split(",")
                        if len(sentiment_parts) == 2:
                            sentiment = sentiment_parts[0].strip()
                            confidence_str = sentiment_parts[1].strip()
                            
                            # Convert confidence to float
                            try:
                                confidence = float(confidence_str)
                                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                            except ValueError:
                                logger.warning(f"Invalid confidence value: {confidence_str}")
                                confidence = 0.0
                            
                            return {
                                "PostId": parsed_id,
                                "Sentiment": sentiment,
                                "Confidence": confidence
                            }
                except Exception as e:
                    logger.warning(f"Error parsing line '{line}': {e}")
                    continue
        
        # Fallback: try to extract sentiment without confidence
        sentiment_keywords = ["Positive", "Negative", "Neutral"]
        for keyword in sentiment_keywords:
            if keyword.lower() in response.lower():
                return {
                    "PostId": post_id,
                    "Sentiment": keyword,
                    "Confidence": 0.0  # Default low confidence for fallback
                }
        
        logger.warning(f"Could not parse confidence response: {response}")
        return None

    @staticmethod
    def parse_batch_responses_with_confidence(
        responses: List[str], 
        post_ids: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Parse a batch of LLM responses with confidence
        """
        parsed_results = []
        
        for response, post_id in zip(responses, post_ids):
            parsed = ResponseParser.parse_sentiment_with_confidence_response(response, post_id)
            if parsed:
                parsed_results.append(parsed)
        
        return parsed_results