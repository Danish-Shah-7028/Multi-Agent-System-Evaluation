"""
System C: Multi-Agent LLM System with Cross-Task Memory
Same as System B (Planner-Executor-Reviewer) but with cross-task learning and memory.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from .groq_client import GroqClient
from .memory_store import MemoryStore
from .system_b import SystemB

logger = logging.getLogger(__name__)

@dataclass
class SystemCResult:
    """Result from System C execution"""
    success_status: bool
    output: str
    steps_taken: int
    total_tokens: int
    api_calls: int
    intermediate_outputs: List[Dict[str, Any]]
    errors: List[str]
    memory_context_used: str = ""
    memory_injections: int = 0
    system: str = "C"

class SystemC:
    """Multi-Agent System with Cross-Task Memory"""
    
    def __init__(self, groq_client: GroqClient, memory_store: MemoryStore):
        self.client = groq_client
        self.memory = memory_store
        self.system_name = "C"
        
        # Use System B as base, but enhance with memory
        self.base_system = SystemB(groq_client)
    
    def solve(self, task_id: int, task: Dict[str, Any]) -> SystemCResult:
        """
        Solve task using multi-agent approach WITH cross-task memory.
        
        Process:
        1. Retrieve relevant memory for this category
        2. Planner: Break task into steps (enhanced with memory context)
        3. Executor: Execute steps (using learned patterns)
        4. Reviewer: Validate (enhanced with successful strategies)
        5. Learn: Update memory with new insights
        """
        logger.info(f"[Task {task_id}] System C starting - Multi-Agent + Memory approach")
        
        category = task.get("category", "general")
        prompt = task.get("prompt", "")
        expected_steps = task.get("expected_steps", 3)
        
        if not prompt:
            return SystemCResult(
                success_status=False,
                output="",
                steps_taken=0,
                total_tokens=0,
                api_calls=0,
                intermediate_outputs=[],
                errors=["Empty task prompt"]
            )
        
        # Retrieve relevant memory context
        memory_context = self.memory.get_memory_context(category)
        memory_injections = 1 if memory_context else 0
        
        intermediate_outputs = []
        total_tokens = 0
        api_calls = 0
        errors = []
        
        # STEP 1: Enhanced Planner (with memory)
        logger.debug(f"[Task {task_id}] Phase 1: Planning (memory-enhanced)")
        enhanced_planner_prompt = self._enhance_with_memory(
            self.base_system._get_planner_prompt(prompt),
            memory_context
        )
        
        plan_response = self.client.call(
            prompt=enhanced_planner_prompt,
            system_prompt=self.base_system._get_planner_system_prompt(),
            max_tokens=512,
            task_id=task_id
        )
        api_calls += 1
        total_tokens += plan_response.tokens_used
        
        if plan_response.error:
            errors.append(f"Planner error: {plan_response.error}")
            return SystemCResult(
                success_status=False,
                output="",
                steps_taken=0,
                total_tokens=total_tokens,
                api_calls=api_calls,
                intermediate_outputs=intermediate_outputs,
                errors=errors,
                memory_context_used=memory_context[:100] if memory_context else "",
                memory_injections=memory_injections
            )
        
        plan_content = plan_response.content
        intermediate_outputs.append({
            "phase": 1,
            "agent": "Planner (Memory-Enhanced)",
            "tokens": plan_response.tokens_used,
            "memory_context_used": bool(memory_context),
            "content": plan_content[:300] + "..." if len(plan_content) > 300 else plan_content
        })
        
        logger.debug(f"[Task {task_id}] Enhanced plan created")
        
        # STEP 2: Executor (using learned patterns)
        logger.debug(f"[Task {task_id}] Phase 2: Execution (pattern-aware)")
        execution_results = []
        
        step_limit = min(expected_steps, 2)
        for step_num in range(1, step_limit + 1):
            # Get successful strategies for this category
            strategies = self.memory.get_successful_strategies(category)
            strategies_context = "\n".join([
                f"- {s['strategy']}: {s['result']}" for s in strategies[:2]
            ]) if strategies else ""
            
            executor_prompt = self.base_system._get_executor_prompt(
                original_task=prompt,
                plan=plan_content,
                step_number=step_num,
                previous_results=execution_results
            )
            
            if strategies_context:
                executor_prompt = self._enhance_with_memory(
                    executor_prompt,
                    f"APPLICABLE SUCCESSFUL STRATEGIES:\n{strategies_context}"
                )
                memory_injections += 1
            
            exec_response = self.client.call(
                prompt=executor_prompt,
                system_prompt=self.base_system._get_executor_system_prompt(),
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
            
            if self.base_system._is_completion_signal(exec_content):
                logger.debug(f"[Task {task_id}] Executor signaled completion at step {step_num}")
                break
        
        execution_summary = "\n".join([
            f"Step {r['step']}: {r['result']}" for r in execution_results
        ])
        
        # STEP 3: Enhanced Reviewer (with proven strategies)
        logger.debug(f"[Task {task_id}] Phase 3: Review (strategy-informed)")
        review_prompt = self.base_system._get_reviewer_prompt(
            original_task=prompt,
            execution_summary=execution_summary
        )
        
        # Inject insights about successful similar problems
        insights = self.memory.get_insights_for_category(category, limit=2)
        if insights:
            insights_text = "\n".join([f"- {i}" for i in insights])
            review_prompt = self._enhance_with_memory(
                review_prompt,
                f"LEARNED INSIGHTS FROM SIMILAR TASKS:\n{insights_text}"
            )
            memory_injections += 1
        
        review_response = self.client.call(
            prompt=review_prompt,
            system_prompt=self.base_system._get_reviewer_system_prompt(),
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
                "agent": "Reviewer (Insight-Enhanced)",
                "tokens": review_response.tokens_used,
                "content": final_output[:300] + "..." if len(final_output) > 300 else final_output
            })
        
        # STEP 4: Learn from this task
        success = len(errors) == 0 and len(final_output) > 0
        self.memory.record_task_completion(
            task_id=task_id,
            category=category,
            success=success,
            output=final_output,
            learnings=self._extract_learnings(task, final_output, success)
        )
        
        # Record strategy if successful
        if success and execution_results:
            strategy = f"Completed in {len(execution_results)} steps using plan-based execution"
            self.memory.add_successful_strategy(
                category=category,
                strategy=strategy,
                result="Successfully solved task"
            )
        
        self.memory.save()
        
        logger.info(f"[Task {task_id}] System C completed - Steps: {len(execution_results)}, Tokens: {total_tokens}, Memory injections: {memory_injections}")
        
        return SystemCResult(
            success_status=success,
            output=final_output,
            steps_taken=len(execution_results),
            total_tokens=total_tokens,
            api_calls=api_calls,
            intermediate_outputs=intermediate_outputs,
            errors=errors,
            memory_context_used=memory_context[:100] if memory_context else "",
            memory_injections=memory_injections
        )
    
    def _enhance_with_memory(self, prompt: str, memory_context: str) -> str:
        """Inject memory context into a prompt"""
        if not memory_context:
            return prompt

        memory_context = memory_context[:1200]
        
        return f"""{prompt}

---
CROSS-TASK MEMORY & LEARNED PATTERNS:
{memory_context}
---

Leverage these learned insights and patterns when solving this task."""
    
    def _extract_learnings(self, task: Dict[str, Any], output: str, success: bool) -> str:
        """Extract key learning from task completion"""
        category = task.get("category", "")
        if not success:
            return f"Failed to solve {category} task - approach needs refinement"
        
        # Generate simple learning
        return f"Successfully applied {category} solving technique with plan-based execution"
