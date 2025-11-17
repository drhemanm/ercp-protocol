"""
Stabilize Operator (O_stab) - Real ML Implementation
Uses sentence transformers for semantic similarity measurement.
"""

from typing import Dict, Any, Optional
from .base import BaseOperator
from server.models.model_registry import model_registry
from sentence_transformers import util


class StabilizeOperator(BaseOperator):
    """
    Check semantic stability between reasoning iterations.

    Features:
    - Semantic similarity using sentence transformers
    - Combined with zero-errors condition
    - Cosine similarity scoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = model_registry.get_sentence_transformer()

    def execute(
        self,
        prev_reasoning: Optional[str],
        curr_reasoning: str,
        threshold: float,
        errors: list,
    ) -> Dict[str, Any]:
        """
        Check if reasoning has stabilized.

        Args:
            prev_reasoning: Previous iteration reasoning (None for first iteration)
            curr_reasoning: Current iteration reasoning
            threshold: Similarity threshold for stability
            errors: List of current errors

        Returns:
            Dictionary with stable flag, score, and metadata
        """
        # First iteration - not stable
        if prev_reasoning is None:
            return {
                "stable": False,
                "score": 0.0,
                "error_count": len(errors),
                "similarity_threshold": threshold,
                "reason": "first_iteration",
            }

        # Compute semantic similarity
        sim_score = self._compute_similarity(prev_reasoning, curr_reasoning)

        # Stable if:
        # 1. High semantic similarity (>= threshold)
        # 2. No errors detected
        stable = (sim_score >= threshold) and (len(errors) == 0)

        result = {
            "stable": stable,
            "score": sim_score,
            "error_count": len(errors),
            "similarity_threshold": threshold,
        }

        # Add reason for instability
        if not stable:
            if sim_score < threshold:
                result["reason"] = "low_similarity"
            elif len(errors) > 0:
                result["reason"] = "has_errors"
        else:
            result["reason"] = "converged"

        return result

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Encode both texts
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.cos_sim(emb1, emb2).item()

        return similarity
