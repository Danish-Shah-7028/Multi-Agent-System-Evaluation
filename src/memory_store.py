"""
Memory store for cross-task learning (System C).
Maintains context from previous tasks and shares with new ones.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MemoryStore:
    """
    Persistent memory for System C.
    Stores key insights, patterns, and successful strategies across tasks.
    """
    
    def __init__(self, memory_file: str = "results/memory_store.json"):
        self.memory_file = memory_file
        self.memory: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "tasks_processed": 0,
            "insights": [],
            "patterns": {},
            "successful_strategies": [],
            "failed_approaches": [],
            "category_knowledge": {}
        }
        self._load_or_init()
    
    def _load_or_init(self):
        """Load existing memory or initialize fresh"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memory = json.load(f)
                logger.info(f"Loaded memory from {self.memory_file}")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}. Starting fresh.")
                self._reset()
        else:
            os.makedirs(os.path.dirname(self.memory_file) or '.', exist_ok=True)
            self._reset()
    
    def _reset(self):
        """Reset memory to initial state"""
        self.memory = {
            "created_at": datetime.now().isoformat(),
            "tasks_processed": 0,
            "insights": [],
            "patterns": {},
            "successful_strategies": [],
            "failed_approaches": [],
            "category_knowledge": {}
        }
    
    def save(self):
        """Persist memory to disk"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
            logger.debug(f"Memory saved to {self.memory_file}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_insight(self, task_id: int, category: str, insight: str):
        """Add a key insight from task completion"""
        self.memory["insights"].append({
            "task_id": task_id,
            "category": category,
            "insight": insight,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"Added insight from task {task_id}: {insight[:50]}...")
    
    def add_pattern(self, category: str, pattern: str, description: str):
        """Learn a pattern from a category"""
        if category not in self.memory["patterns"]:
            self.memory["patterns"][category] = []
        
        self.memory["patterns"][category].append({
            "pattern": pattern,
            "description": description,
            "learned_at": datetime.now().isoformat()
        })
        logger.debug(f"Learned pattern for {category}: {pattern}")
    
    def add_successful_strategy(self, category: str, strategy: str, result: str):
        """Record a successful strategy"""
        self.memory["successful_strategies"].append({
            "category": category,
            "strategy": strategy,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"Recorded successful strategy for {category}")
    
    def add_failed_approach(self, task_id: int, category: str, approach: str, reason: str):
        """Record what didn't work (negative learning)"""
        self.memory["failed_approaches"].append({
            "task_id": task_id,
            "category": category,
            "approach": approach,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"Recorded failed approach: {reason[:50]}...")
    
    def update_category_knowledge(self, category: str, key: str, value: Any):
        """Update knowledge for a specific category"""
        if category not in self.memory["category_knowledge"]:
            self.memory["category_knowledge"][category] = {}
        
        self.memory["category_knowledge"][category][key] = value
        logger.debug(f"Updated {category} knowledge: {key}")
    
    def get_insights_for_category(self, category: str, limit: int = 5) -> List[str]:
        """Retrieve key insights for a category"""
        category_insights = [
            i["insight"] for i in self.memory["insights"]
            if i["category"] == category
        ]
        return category_insights[-limit:]
    
    def get_patterns_for_category(self, category: str) -> List[Dict[str, str]]:
        """Retrieve learned patterns for a category"""
        return self.memory["patterns"].get(category, [])
    
    def get_successful_strategies(self, category: str) -> List[Dict[str, str]]:
        """Retrieve successful strategies for a category"""
        return [
            s for s in self.memory["successful_strategies"]
            if s["category"] == category
        ]
    
    def get_memory_context(self, category: str) -> str:
        """Get formatted memory context for prompt injection"""
        insights = self.get_insights_for_category(category, limit=3)
        patterns = self.get_patterns_for_category(category)
        strategies = self.get_successful_strategies(category)
        
        context_parts = []
        
        if insights:
            context_parts.append("## Previous Insights:")
            for insight in insights:
                context_parts.append(f"- {insight}")
        
        if patterns:
            context_parts.append("\n## Learned Patterns:")
            for p in patterns:
                context_parts.append(f"- {p['pattern']}: {p['description']}")
        
        if strategies:
            context_parts.append("\n## Successful Strategies:")
            for s in strategies:
                context_parts.append(f"- {s['strategy']}: {s['result']}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def record_task_completion(
        self,
        task_id: int,
        category: str,
        success: bool,
        output: str,
        learnings: Optional[str] = None
    ):
        """Record completion of a task and any learnings"""
        self.memory["tasks_processed"] += 1
        
        if learnings:
            self.add_insight(task_id, category, learnings)
        
        # Update category knowledge with success rate
        if category not in self.memory["category_knowledge"]:
            self.memory["category_knowledge"][category] = {
                "success_count": 0,
                "attempt_count": 0
            }
        
        self.memory["category_knowledge"][category]["attempt_count"] += 1
        if success:
            self.memory["category_knowledge"][category]["success_count"] += 1
        
        logger.info(f"Task {task_id} ({category}) - Success: {success}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary statistics"""
        total_insights = len(self.memory["insights"])
        total_patterns = sum(len(p) for p in self.memory["patterns"].values())
        total_strategies = len(self.memory["successful_strategies"])
        total_failed = len(self.memory["failed_approaches"])
        
        success_stats = {}
        for cat, knowledge in self.memory["category_knowledge"].items():
            attempts = knowledge.get("attempt_count", 0)
            successes = knowledge.get("success_count", 0)
            success_rate = (successes / attempts * 100) if attempts > 0 else 0
            success_stats[cat] = {
                "attempts": attempts,
                "successes": successes,
                "success_rate": f"{success_rate:.1f}%"
            }
        
        return {
            "tasks_processed": self.memory["tasks_processed"],
            "total_insights": total_insights,
            "total_patterns": total_patterns,
            "total_strategies": total_strategies,
            "failed_approaches": total_failed,
            "category_success_rates": success_stats
        }
    
    def clear(self):
        """Clear all memory (for fresh start)"""
        self._reset()
        self.save()
        logger.info("Memory cleared")
