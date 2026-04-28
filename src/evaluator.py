"""
Semantic similarity evaluation for experiment scoring.
Uses a lightweight local heuristic to avoid consuming API quota during large runs.
"""

import json
import re
import string
import logging
from difflib import SequenceMatcher
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from .groq_client import GroqClient

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of evaluating a task output"""
    task_id: int
    system: str
    success: int  # 1 or 0
    similarity_score: float  # 0.0 to 1.0
    error_category: Optional[str]  # None, "logical_error", "incomplete_output", "hallucination"
    reasoning: str
    tokens_used: int

class SemanticEvaluator:
    """Evaluate task outputs using a lightweight local similarity heuristic"""
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
        self.threshold = 0.8  # Success threshold
    
    def evaluate(
        self,
        task_id: int,
        system: str,
        reference_answer: str,
        generated_output: str,
        task_description: str = ""
    ) -> EvaluationResult:
        """
        Evaluate if generated output matches reference answer
        
        Args:
            task_id: Task identifier
            system: System name (A, B, or C)
            reference_answer: Expected correct answer
            generated_output: Output from system
            task_description: Optional task description for context
        
        Returns:
            EvaluationResult with similarity score and error category
        """
        if not generated_output or not generated_output.strip():
            return EvaluationResult(
                task_id=task_id,
                system=system,
                success=0,
                similarity_score=0.0,
                error_category="incomplete_output",
                reasoning="Empty or no output generated",
                tokens_used=0
            )
        
        score, error_cat, reasoning = self._score_locally(
            reference_answer=reference_answer,
            generated_output=generated_output,
            task_description=task_description
        )

        success = 1 if score >= self.threshold else 0

        return EvaluationResult(
            task_id=task_id,
            system=system,
            success=success,
            similarity_score=score,
            error_category=error_cat if success == 0 else None,
            reasoning=reasoning,
            tokens_used=0
        )

    def _score_locally(
        self,
        reference_answer: str,
        generated_output: str,
        task_description: str = ""
    ) -> Tuple[float, Optional[str], str]:
        reference_norm = self._normalize_text(reference_answer)
        generated_norm = self._normalize_text(generated_output)

        if not generated_norm:
            return 0.0, "incomplete_output", "Empty or no output generated"

        seq_score = SequenceMatcher(None, reference_norm, generated_norm).ratio()
        reference_tokens = set(reference_norm.split())
        generated_tokens = set(generated_norm.split())
        if reference_tokens:
            overlap = len(reference_tokens & generated_tokens) / len(reference_tokens)
        else:
            overlap = 0.0

        score = round((0.6 * seq_score) + (0.4 * overlap), 3)

        if score < 0.35:
            error_category = "hallucination" if self._looks_hallucinated(generated_norm) else "logical_error"
        elif score < 0.7:
            error_category = "incomplete_output"
        else:
            error_category = None

        reasoning = (
            f"Local similarity score={score:.3f} using sequence overlap and word overlap. "
            f"Task context={'present' if task_description else 'absent'}."
        )
        return score, error_category, reasoning

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _looks_hallucinated(self, text: str) -> bool:
        markers = ["as an ai", "made up", "fictional", "cannot verify", "not in the reference"]
        return any(marker in text for marker in markers)
    
    def batch_evaluate(
        self,
        task_id: int,
        reference_answer: str,
        results: Dict[str, str]  # {system: output}
    ) -> Dict[str, EvaluationResult]:
        """Evaluate all systems for a single task"""
        evaluations = {}
        for system, output in results.items():
            eval_result = self.evaluate(
                task_id=task_id,
                system=system,
                reference_answer=reference_answer,
                generated_output=output
            )
            evaluations[system] = eval_result
        return evaluations
