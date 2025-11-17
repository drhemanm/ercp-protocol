"""
Stabilize Operator - Semantic Stability Checker
Author: ERCP Protocol Implementation
License: Apache-2.0

Implements the Stabilize operator (O_stab) from the ERCP protocol.
Checks semantic similarity between reasoning iterations.
"""

from typing import Dict, Any, Optional, List

from .base import BaseOperator
from ..models.model_registry import get_model_registry


class StabilizeOperator(BaseOperator):
    """
    Check semantic stability between reasoning iterations.

    This operator:
    1. Accepts current and previous reasoning text
    2. Computes semantic similarity using sentence-transformers
    3. Uses embedding model (e.g., 'all-MiniLM-L6-v2')
    4. Computes cosine similarity between embeddings
    5. Checks stability condition: (similarity >= threshold) AND (no errors)
    6. Returns {stable: bool, score: float, error_count: int}
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs
    ):
        """
        Initialize the Stabilize operator.

        Args:
            embedding_model: Name of the sentence-transformer model to use
            **kwargs: Additional arguments passed to BaseOperator
        """
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.registry = get_model_registry()

    def _compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Get embedding model
            model = self.registry.get_embedding_model(self.embedding_model)

            # Encode texts
            embeddings = model.encode([text1, text2])

            # Compute cosine similarity
            from sentence_transformers import util
            similarity = util.cos_sim(embeddings[0], embeddings[1])

            # Extract scalar value
            similarity_score = float(similarity[0][0])

            self.logger.debug(f"Similarity score: {similarity_score:.4f}")

            return similarity_score

        except Exception as e:
            self.logger.error(f"Similarity computation failed: {str(e)}")
            # Return 0 on error (conservative approach)
            return 0.0

    def _compute_edit_distance(self, text1: str, text2: str) -> float:
        """
        Compute normalized edit distance as fallback similarity metric.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Normalized similarity score (0.0 to 1.0)
        """
        # Simple Levenshtein distance implementation
        len1, len2 = len(text1), len(text2)

        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0

        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if text1[i-1] == text2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + 1     # substitution
                    )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)

        # Normalize to 0-1 range (1 = identical, 0 = completely different)
        similarity = 1.0 - (distance / max_len)

        return similarity

    def execute(
        self,
        reasoning_curr: str,
        reasoning_prev: Optional[str] = None,
        threshold: float = 0.95,
        errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the Stabilize operator.

        Args:
            reasoning_curr: Current iteration reasoning text
            reasoning_prev: Previous iteration reasoning text (None for first iteration)
            threshold: Similarity threshold for stability
            errors: List of errors from verification (if any)
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
                - stable: Boolean indicating if reasoning is stable
                - score: Similarity score
                - error_count: Number of errors (if errors provided)
                - similarity_threshold: The threshold used
        """
        if errors is None:
            errors = []

        # First iteration is always unstable
        if reasoning_prev is None:
            self.logger.info("First iteration - marking as unstable")
            return {
                "stable": False,
                "score": 0.0,
                "error_count": len(errors),
                "similarity_threshold": threshold
            }

        self.logger.info(
            f"Computing stability (threshold: {threshold}, errors: {len(errors)})"
        )

        # Compute semantic similarity
        similarity = self._compute_similarity(reasoning_prev, reasoning_curr)

        # Stability condition: high similarity AND no errors
        is_stable = (similarity >= threshold) and (len(errors) == 0)

        self.logger.info(
            f"Stability check: score={similarity:.4f}, "
            f"errors={len(errors)}, stable={is_stable}"
        )

        return {
            "stable": is_stable,
            "score": similarity,
            "error_count": len(errors),
            "similarity_threshold": threshold
        }
