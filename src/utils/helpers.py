"""
Utility functions and helpers
"""

import os
import logging
import yaml
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_file = config.get('file', 'logs/experiment.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.get('level', 'INFO')),
        format=config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return {}

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if necessary"""
    os.makedirs(path, exist_ok=True)
