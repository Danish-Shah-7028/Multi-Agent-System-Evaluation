"""
Main experiment runner orchestrating all systems and evaluations.
"""

import json
import csv
import logging
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from .config import get_config
from .groq_client import GroqClient
from .evaluator import SemanticEvaluator
from .memory_store import MemoryStore
from .system_a import SystemA
from .system_b import SystemB
from .system_c import SystemC

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Orchestrates the full research study"""
    
    def __init__(self, config_path: str = "config.yaml", data_path: str = "data/tasks_dataset.json"):
        self.config = get_config(config_path)
        self.data_path = data_path
        
        # Initialize components
        self.groq_client = GroqClient(config_path=config_path)
        self.evaluator = SemanticEvaluator(self.groq_client)
        self.memory_store = MemoryStore("results/memory_store.json")
        
        # Initialize systems
        self.system_a = SystemA(self.groq_client)
        self.system_b = SystemB(self.groq_client)
        self.system_c = SystemC(self.groq_client, self.memory_store)
        
        # Results storage
        self.results = []  # List of result dicts
        self.raw_logs = {}  # Detailed logs for sample tasks
        
        # Load tasks
        self.tasks = self._load_tasks()
        logger.info(f"Loaded {len(self.tasks)} tasks from {data_path}")
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load task dataset"""
        if not os.path.exists(self.data_path):
            logger.error(f"Task file not found: {self.data_path}")
            return []
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data.get("tasks", [])
    
    def run_all_experiments(self, task_limit: int = None) -> Dict[str, Any]:
        """
        Run all systems on all tasks.
        
        Args:
            task_limit: Limit number of tasks (for testing)
        
        Returns:
            Summary statistics
        """
        tasks = self.tasks[:task_limit] if task_limit else self.tasks
        logger.info(f"Starting experiment run on {len(tasks)} tasks")
        
        for idx, task in enumerate(tasks, 1):
            task_id = task.get("id", idx)
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {task_id}/{len(tasks)}: {task.get('title', 'Untitled')}")
            logger.info(f"Category: {task.get('category', 'unknown')}")
            logger.info(f"{'='*60}")
            
            # Run all three systems
            system_outputs = {}
            system_steps = {}
            system_tokens = {}
            system_logs = {}
            
            # System A
            try:
                logger.info("Running System A (Single-Agent)...")
                result_a = self.system_a.solve(task_id, task)
                system_outputs["A"] = result_a.output
                system_steps["A"] = result_a.steps_taken
                system_tokens["A"] = result_a.total_tokens
                system_logs["A"] = {
                    "api_calls": result_a.api_calls,
                    "errors": result_a.errors,
                    "intermediate": result_a.intermediate_outputs
                }
            except Exception as e:
                logger.error(f"System A failed: {e}")
                system_outputs["A"] = ""
                system_steps["A"] = 0
                system_tokens["A"] = 0
                system_logs["A"] = {"error": str(e)}
            
            # System B
            try:
                logger.info("Running System B (Multi-Agent)...")
                result_b = self.system_b.solve(task_id, task)
                system_outputs["B"] = result_b.output
                system_steps["B"] = result_b.steps_taken
                system_tokens["B"] = result_b.total_tokens
                system_logs["B"] = {
                    "api_calls": result_b.api_calls,
                    "errors": result_b.errors,
                    "intermediate": result_b.intermediate_outputs
                }
            except Exception as e:
                logger.error(f"System B failed: {e}")
                system_outputs["B"] = ""
                system_steps["B"] = 0
                system_tokens["B"] = 0
                system_logs["B"] = {"error": str(e)}
            
            # System C
            try:
                logger.info("Running System C (Multi-Agent + Memory)...")
                result_c = self.system_c.solve(task_id, task)
                system_outputs["C"] = result_c.output
                system_steps["C"] = result_c.steps_taken
                system_tokens["C"] = result_c.total_tokens
                system_logs["C"] = {
                    "api_calls": result_c.api_calls,
                    "errors": result_c.errors,
                    "intermediate": result_c.intermediate_outputs,
                    "memory_injections": result_c.memory_injections
                }
            except Exception as e:
                logger.error(f"System C failed: {e}")
                system_outputs["C"] = ""
                system_steps["C"] = 0
                system_tokens["C"] = 0
                system_logs["C"] = {"error": str(e)}
            
            # Evaluate all systems
            reference_answer = task.get("reference_answer", "")
            evaluations = self.evaluator.batch_evaluate(
                task_id=task_id,
                reference_answer=reference_answer,
                results=system_outputs
            )
            
            # Store results
            for system in ["A", "B", "C"]:
                eval_result = evaluations.get(system)
                if eval_result:
                    result_row = {
                        "task_id": task_id,
                        "task_title": task.get("title", ""),
                        "category": task.get("category", ""),
                        "system": system,
                        "success": eval_result.success,
                        "similarity_score": round(eval_result.similarity_score, 3),
                        "steps": system_steps.get(system, 0),
                        "tokens": system_tokens.get(system, 0),
                        "error_type": eval_result.error_category or "none",
                        "reasoning": eval_result.reasoning[:100] if eval_result.reasoning else ""
                    }
                    self.results.append(result_row)
                    
                    logger.info(f"  System {system}: Success={eval_result.success}, Score={eval_result.similarity_score:.2f}, Steps={system_steps.get(system, 0)}, Tokens={system_tokens.get(system, 0)}")
            
            # Store detailed logs for first 5 tasks
            if task_id <= 5:
                self.raw_logs[task_id] = {
                    "task": {
                        "id": task_id,
                        "title": task.get("title", ""),
                        "prompt": task.get("prompt", "")[:200],
                        "reference_answer": task.get("reference_answer", "")[:200]
                    },
                    "systems": system_logs
                }
        
        logger.info(f"\n{'='*60}")
        logger.info("All experiments completed!")
        logger.info(f"{'='*60}\n")
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results:
            return {}
        
        summary = {
            "total_experiments": len(self.results),
            "total_tasks": len(set(r["task_id"] for r in self.results)),
            "timestamp": datetime.now().isoformat(),
            "by_system": {}
        }
        
        for system in ["A", "B", "C"]:
            system_results = [r for r in self.results if r["system"] == system]
            if system_results:
                successes = sum(r["success"] for r in system_results)
                summary["by_system"][system] = {
                    "name": f"System {system}",
                    "total_tasks": len(system_results),
                    "successes": successes,
                    "success_rate": round(successes / len(system_results) * 100, 1),
                    "avg_steps": round(sum(r["steps"] for r in system_results) / len(system_results), 1),
                    "avg_tokens": round(sum(r["tokens"] for r in system_results) / len(system_results), 1),
                    "total_tokens": sum(r["tokens"] for r in system_results),
                    "avg_similarity": round(sum(r["similarity_score"] for r in system_results) / len(system_results), 3)
                }
        
        return summary
    
    def save_results(self):
        """Save all results to files"""
        os.makedirs("results", exist_ok=True)
        
        # Save as JSON
        logger.info("Saving results as JSON...")
        with open("results/experiment_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Saved to results/experiment_results.json")
        
        # Save as CSV
        logger.info("Saving results as CSV...")
        if self.results:
            with open("results/experiment_results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            logger.info("Saved to results/experiment_results.csv")
        
        # Save raw logs
        logger.info("Saving raw logs...")
        with open("results/raw_logs.json", "w") as f:
            json.dump(self.raw_logs, f, indent=2)
        logger.info("Saved to results/raw_logs.json")
        
        # Save memory summary
        logger.info("Saving memory summary...")
        with open("results/memory_summary.json", "w") as f:
            json.dump(self.memory_store.get_summary(), f, indent=2)
        logger.info("Saved to results/memory_summary.json")
        
        logger.info("All results saved!")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results"""
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return self._generate_summary()
    
    def clear_memory(self):
        """Clear System C memory for fresh start"""
        self.memory_store.clear()
        logger.info("Memory cleared")
