"""
Multi-Agent vs Single-Agent LLM Research Study
An empirical study comparing system architectures for task automation
"""

__version__ = "1.0.0"
__author__ = "AI Research Lab"

from .config import get_config, Config
from .groq_client import GroqClient, APIResponse
from .evaluator import SemanticEvaluator, EvaluationResult
from .memory_store import MemoryStore
from .system_a import SystemA
from .system_b import SystemB
from .system_c import SystemC
from .experiment_runner import ExperimentRunner

__all__ = [
    "get_config",
    "Config",
    "GroqClient",
    "APIResponse",
    "SemanticEvaluator",
    "EvaluationResult",
    "MemoryStore",
    "SystemA",
    "SystemB",
    "SystemC",
    "ExperimentRunner"
]
