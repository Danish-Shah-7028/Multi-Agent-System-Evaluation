"""
System B: Multi-Agent LLM System (Planner-Executor-Reviewer)
Three specialized agents: Planner breaks down task, Executor performs steps, Reviewer validates.
"""

import json
import re
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from .groq_client import GroqClient

logger = logging.getLogger(__name__)

@dataclass
class SystemBResult:
    """Result from System B execution"""
    success_status: bool
    output: str
    steps_taken: int
    total_tokens: int
    api_calls: int
    intermediate_outputs: List[Dict[str, Any]]
    errors: List[str]
    system: str = "B"

class SystemB:
    """Multi-Agent System: Planner → Executor → Reviewer"""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
        self.system_name = "B"
    
    def solve(self, task_id: int, task: Dict[str, Any]) -> SystemBResult:
        """
        Solve task using multi-agent approach.
        
        Process:
        1. Planner: Break task into steps
        2. Executor: Execute each step
        3. Reviewer: Validate and refine output
        """
        logger.info(f"[Task {task_id}] System B starting - Multi-Agent approach")
        
        intermediate_outputs = []
        total_tokens = 0
        api_calls = 0
        errors = []
        
        prompt = task.get("prompt", "")
        expected_steps = task.get("expected_steps", 3)
        
        if not prompt:
            return SystemBResult(
                success_status=False,
                output="",
                steps_taken=0,
                total_tokens=0,
                api_calls=0,
                intermediate_outputs=[],
                errors=["Empty task prompt"]
            )
        
        # STEP 1: Planner - Break down the task
        logger.debug(f"[Task {task_id}] Phase 1: Planning")
        plan_response = self.client.call(
            prompt=self._get_planner_prompt(prompt),
            system_prompt=self._get_planner_system_prompt(),
            max_tokens=512,
            task_id=task_id
        )
        api_calls += 1
        total_tokens += plan_response.tokens_used
        
        if plan_response.error:
            errors.append(f"Planner error: {plan_response.error}")
            return SystemBResult(
                success_status=False,
                output="",
                steps_taken=0,
                total_tokens=total_tokens,
                api_calls=api_calls,
                intermediate_outputs=intermediate_outputs,
                errors=errors
            )
        
        plan_content = plan_response.content
        intermediate_outputs.append({
            "phase": 1,
            "agent": "Planner",
            "tokens": plan_response.tokens_used,
            "content": plan_content[:300] + "..." if len(plan_content) > 300 else plan_content
        })
        
        logger.debug(f"[Task {task_id}] Plan created: {plan_content[:100]}...")
        
        # STEP 2: Executor - Execute the plan
        logger.debug(f"[Task {task_id}] Phase 2: Execution")
        execution_results = []
        
        step_limit = min(expected_steps, 2)  # Hard cap to keep request count under the daily limit
        for step_num in range(1, step_limit + 1):
            executor_prompt = self._get_executor_prompt(
                original_task=prompt,
                plan=plan_content,
                step_number=step_num,
                previous_results=execution_results
            )
            
            exec_response = self.client.call(
                prompt=executor_prompt,
                system_prompt=self._get_executor_system_prompt(),
                max_tokens=768,
                task_id=task_id
            )
            api_calls += 1
            total_tokens += exec_response.tokens_used
            
            if exec_response.error:
                errors.append(f"Executor error at step {step_num}: {exec_response.error}")
                break
            
            exec_content = exec_response.content.strip()
            execution_results.append({
                "step": step_num,
                "result": exec_content
            })
            
            intermediate_outputs.append({
                "phase": 2,
                "agent": "Executor",
                "step": step_num,
                "tokens": exec_response.tokens_used,
                "content": exec_content[:200] + "..." if len(exec_content) > 200 else exec_content
            })
            
            # Check if executor indicates task completion
            if self._is_completion_signal(exec_content):
                logger.debug(f"[Task {task_id}] Executor signaled completion at step {step_num}")
                break
        
        # Compile execution results
        execution_summary = "\n".join([
            f"Step {r['step']}: {r['result']}" for r in execution_results
        ])
        
        # STEP 3: Reviewer - Validate and refine
        logger.debug(f"[Task {task_id}] Phase 3: Review")
        review_prompt = self._get_reviewer_prompt(
            original_task=prompt,
            execution_summary=execution_summary
        )
        
        review_response = self.client.call(
            prompt=review_prompt,
            system_prompt=self._get_reviewer_system_prompt(),
            max_tokens=768,
            task_id=task_id
        )
        api_calls += 1
        total_tokens += review_response.tokens_used
        
        if review_response.error:
            errors.append(f"Reviewer error: {review_response.error}")
            final_output = execution_summary
        else:
            final_output = review_response.content.strip()
            intermediate_outputs.append({
                "phase": 3,
                "agent": "Reviewer",
                "tokens": review_response.tokens_used,
                "content": final_output[:300] + "..." if len(final_output) > 300 else final_output
            })
        
        logger.info(f"[Task {task_id}] System B completed - Steps: {len(execution_results)}, Tokens: {total_tokens}, Calls: {api_calls}")
        
        return SystemBResult(
            success_status=len(errors) == 0 and len(final_output) > 0,
            output=final_output,
            steps_taken=len(execution_results),
            total_tokens=total_tokens,
            api_calls=api_calls,
            intermediate_outputs=intermediate_outputs,
            errors=errors
        )
    
    def _is_completion_signal(self, text: str) -> bool:
        """Check if executor indicates task is complete"""
        signals = ["task complete", "done", "final answer", "solution found", "[end]", "[done]"]
        return any(signal in text.lower() for signal in signals)
    
    def _get_planner_system_prompt(self) -> str:
        return """You are a strategic planner. Break the task into a short, practical plan.

    Return only the essential steps, dependencies, and checks in a compact numbered list."""
    
    def _get_planner_prompt(self, task: str) -> str:
        compact_task = task[:1200]

        return f"""Please analyze this task and create a concise execution plan:

    TASK: {compact_task}

Provide:
    1. 3-5 key steps
    2. Success criteria for each step
    3. Any critical edge cases

    Be concise and specific."""
    
    def _get_executor_system_prompt(self) -> str:
        return """You are an executor specialized in solving one step at a time.

    Focus on the current step only, keep the response compact, and write [END] when the task is complete."""
    
    def _get_executor_prompt(
        self,
        original_task: str,
        plan: str,
        step_number: int,
        previous_results: List[Dict[str, str]]
    ) -> str:
        previous_context = ""
        if previous_results:
            previous_context = "\n\nPREVIOUS RESULTS:\n"
            for result in previous_results[-1:]:  # Keep only the latest result to reduce payload size
                previous_context += f"Step {result['step']}: {result['result']}\n"

        compact_task = original_task[:900]
        compact_plan = plan[:900]
        
        return f"""Execute step {step_number} of the following plan:

    ORIGINAL TASK: {compact_task}

    PLAN: {compact_plan}

CURRENT STEP: Step {step_number}{previous_context}

    Execute this specific step thoroughly but briefly. Provide concrete results.
Write [END] if this completes the task."""
    
    def _get_reviewer_system_prompt(self) -> str:
        return """You are a quality reviewer.

    Check correctness, remove redundancy, and return a concise final answer."""
    
    def _get_reviewer_prompt(self, original_task: str, execution_summary: str) -> str:
        compact_task = original_task[:900]
        compact_summary = execution_summary[:1800]

        return f"""Review and synthesize the following execution results:

    ORIGINAL TASK: {compact_task}

    EXECUTION RESULTS:
    {compact_summary}

    Provide a final answer that is correct, complete, and well-structured, but keep it compact."""
