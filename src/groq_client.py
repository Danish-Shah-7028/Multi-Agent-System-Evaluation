"""
Groq API client wrapper for making LLM calls.
Handles rate limiting, retries, and token counting.
"""

import time
import json
import logging
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from .config import get_config

try:
    from groq import Groq
except ImportError:
    raise ImportError("Groq library not installed. Install with: pip install groq")

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Response from Groq API"""
    content: str
    tokens_used: int
    model: str
    stop_reason: str
    latency_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class GroqClient:
    """Wrapper for Groq API with rate limiting, retry logic, and multi-key support"""
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.api_key = api_key or self.config.get_groq_api_key()
        self.available_keys = self.config.get_all_groq_api_keys()
        self.current_key_index = 0
        
        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
        
        self.groq_config = self.config.get_groq_config()
        # Allow overriding model via environment variable for quick switching
        env_model = os.environ.get("GROQ_MODEL")
        if env_model:
            self.model = env_model
        else:
            self.model = self.groq_config.get("model", "mixtral-8x7b-32768")
        self.temperature = self.groq_config.get("temperature", 0.7)
        self.max_tokens = self.groq_config.get("max_tokens", 2048)
        self.timeout = self.groq_config.get("timeout", 30)
        
        self.rate_limit_delay = self.config.get("experiment.rate_limit_delay", 1.0)
        self.max_retries = self.config.get("experiment.max_retries", 3)
        self.last_call_time = 0
        self.total_tokens_used = 0
        self.call_count = 0
    
    def _apply_rate_limit(self):
        """Apply rate limiting between calls"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call_time = time.time()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of tokens (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _rotate_to_next_key(self):
        """Rotate to the next available API key (for handling rate limits)"""
        if len(self.available_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.available_keys)
            self.api_key = self.available_keys[self.current_key_index]
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Rotated to next API key (index {self.current_key_index})")
            except Exception as e:
                logger.error(f"Failed to switch to next API key: {e}")
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_id: Optional[int] = None
    ) -> APIResponse:
        """
        Make a call to Groq API with retry logic
        
        Args:
            prompt: Main prompt/message
            system_prompt: Optional system message for context
            temperature: Optional override for temperature
            max_tokens: Optional override for max tokens
            task_id: Optional task ID for logging
        
        Returns:
            APIResponse with content and metadata
        """
        self._apply_rate_limit()
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    timeout=self.timeout
                )
                
                latency = (time.time() - start_time) * 1000  # ms
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else self._estimate_tokens(content)
                
                self.total_tokens_used += tokens_used
                self.call_count += 1
                
                logger.info(f"[Task {task_id}] API call successful | Tokens: {tokens_used} | Latency: {latency:.0f}ms")
                
                return APIResponse(
                    content=content,
                    tokens_used=tokens_used,
                    model=self.model,
                    stop_reason=response.choices[0].finish_reason or "completed",
                    latency_ms=latency
                )
            
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    error_msg = f"API call failed after {self.max_retries} retries: {str(e)}"
                    logger.error(f"[Task {task_id}] {error_msg}")
                    return APIResponse(
                        content="",
                        tokens_used=0,
                        model=self.model,
                        stop_reason="error",
                        latency_ms=(time.time() - start_time) * 1000,
                        error=error_msg
                    )
                
                wait_time = 2 ** retry_count  # exponential backoff
                logger.warning(f"[Task {task_id}] Retry {retry_count}/{self.max_retries} after {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        return APIResponse(
            content="",
            tokens_used=0,
            model=self.model,
            stop_reason="error",
            latency_ms=(time.time() - start_time) * 1000,
            error="Max retries exceeded"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "total_calls": self.call_count,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_call": self.total_tokens_used / max(1, self.call_count),
            "model": self.model
        }
