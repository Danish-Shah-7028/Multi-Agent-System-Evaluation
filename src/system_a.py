"""
System A: Single-Agent LLM System
One LLM handles the entire task end-to-end.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from .groq_client import GroqClient, APIResponse

logger = logging.getLogger(__name__)

@dataclass
class SystemAResult:
    """Result from System A execution"""
    success_status: bool
    output: str
    steps_taken: int
    total_tokens: int
    api_calls: int
    intermediate_outputs: List[Dict[str, Any]]
    errors: List[str]
    system: str = "A"

class SystemA:
    """Single-Agent System: One LLM solves the entire task"""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
        self.system_name = "A"
    
    def solve(self, task_id: int, task: Dict[str, Any]) -> SystemAResult:
        """
        Solve task with single LLM call.
        
        Args:
            task_id: Task identifier
            task: Task dictionary with 'prompt' and 'reference_answer'
        
        Returns:
            SystemAResult with output and metadata
        """
        logger.info(f"[Task {task_id}] System A starting - Single-Agent approach")
        
        prompt = task.get("prompt", "")
        if not prompt:
            return SystemAResult(
                success_status=False,
                output="",
                steps_taken=0,
                total_tokens=0,
                api_calls=0,
                intermediate_outputs=[],
                errors=["Empty task prompt"]
            )
        
        # System A: Single API call with full task context
        system_prompt = self._get_system_prompt()
        
        response = self.client.call(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            task_id=task_id
        )
        
        if response.error:
            logger.error(f"[Task {task_id}] System A failed: {response.error}")
            return SystemAResult(
                success_status=False,
                output="",
                steps_taken=1,
                total_tokens=response.tokens_used,
                api_calls=1,
                intermediate_outputs=[{
                    "step": 1,
                    "type": "error",
                    "content": response.error
                }],
                errors=[response.error]
            )
        
        output = response.content.strip()
        
        logger.info(f"[Task {task_id}] System A completed - Output length: {len(output)} chars, Tokens: {response.tokens_used}")
        
        return SystemAResult(
            success_status=True,
            output=output,
            steps_taken=1,
            total_tokens=response.tokens_used,
            api_calls=1,
            intermediate_outputs=[{
                "step": 1,
                "type": "single_response",
                "tokens": response.tokens_used,
                "content": output[:200] + "..." if len(output) > 200 else output
            }],
            errors=[]
        )
    
    def _get_system_prompt(self) -> str:
        return """You are an intelligent assistant tasked with solving complex problems.

    Reason carefully, stay concise, and provide a clear final answer with only the necessary explanation."""
