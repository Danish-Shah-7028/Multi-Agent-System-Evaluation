"""
Configuration management for the research study.
Loads settings from config.yaml and provides defaults.
"""

import os
import json
from typing import Dict, Any, Optional, List
import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

class Config:
    """Central configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load .env file if dotenv is available
        if load_dotenv:
            load_dotenv()
        
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
        self._api_key_index = 0  # For round-robin key rotation
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found"""
        return {
            "groq": {
                "model": "mixtral-8x7b-32768",
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 30
            },
            "experiment": {
                "num_tasks": 25,
                "success_threshold": 0.8,
                "rate_limit_delay": 1.0,
                "max_retries": 3
            }
        }
    
    def _setup_directories(self):
        """Ensure output directories exist"""
        dirs = ["data", "results", "outputs", "logs"]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation: 'groq.model'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_all_groq_api_keys(self) -> List[str]:
        """Get all available Groq API keys from numbered slots (1-4) or single GROQ_API_KEY"""
        keys = []
        # Check for numbered keys first
        for i in range(1, 5):
            key = os.environ.get(f"GROQ_API_KEY_{i}")
            if key:
                keys.append(key)
        # Fall back to single GROQ_API_KEY if no numbered keys found
        if not keys:
            key = os.environ.get("GROQ_API_KEY")
            if key:
                keys.append(key)
        return keys
    
    def get_next_groq_api_key(self) -> str:
        """Get next Groq API key in round-robin fashion"""
        keys = self.get_all_groq_api_keys()
        if not keys:
            raise ValueError("No Groq API keys found in environment (GROQ_API_KEY or GROQ_API_KEY_1..4)")
        key = keys[self._api_key_index % len(keys)]
        self._api_key_index += 1
        return key
    
    def get_groq_api_key(self) -> str:
        """Get the first/primary Groq API key from environment"""
        keys = self.get_all_groq_api_keys()
        if not keys:
            raise ValueError("No Groq API keys found in environment (GROQ_API_KEY or GROQ_API_KEY_1..4)")
        return keys[0]
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq API configuration"""
        return self.config.get("groq", {})
    
    def get_systems_config(self) -> Dict[str, Dict[str, Any]]:
        """Get all systems configuration"""
        return self.config.get("systems", {})
    
    def get_task_categories(self) -> list:
        """Get task categories"""
        return self.config.get("task_categories", [])


# Global config instance
_config_instance: Optional[Config] = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

def set_config(config: Config):
    """Set global config instance"""
    global _config_instance
    _config_instance = config
