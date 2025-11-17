"""
Constraint Utilities - Helper Functions for Constraint Management
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides utility functions for:
- Deduplicating constraints
- Merging similar constraints
- Computing semantic similarity
- Sorting by priority
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from ..models.model_registry import get_model_registry


logger = logging.getLogger(__name__)


def constraint_semantic_similarity(
    constraint1: Dict[str, Any],
    constraint2: Dict[str, Any],
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Compute semantic similarity between two constraints.

    Args:
        constraint1: First constraint dictionary
        constraint2: Second constraint dictionary
        model_name: Name of the sentence-transformer model to use

    Returns:
        Similarity score (0.0 to 1.0)
    """
    nl_text1 = constraint1.get("nl_text", "")
    nl_text2 = constraint2.get("nl_text", "")

    if not nl_text1 or not nl_text2:
        return 0.0

    try:
        registry = get_model_registry()
        model = registry.get_embedding_model(model_name)

        # Encode texts
        embeddings = model.encode([nl_text1, nl_text2])

        # Compute cosine similarity
        from sentence_transformers import util
        similarity = util.cos_sim(embeddings[0], embeddings[1])

        # Extract scalar value
        similarity_score = float(similarity[0][0])

        return similarity_score

    except Exception as e:
        logger.error(f"Similarity computation failed: {str(e)}")
        return 0.0


def deduplicate_constraints(
    constraints: List[Dict[str, Any]],
    similarity_threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """
    Deduplicate constraints based on semantic similarity.

    Args:
        constraints: List of constraint dictionaries
        similarity_threshold: Similarity threshold for considering duplicates

    Returns:
        Deduplicated list of constraints
    """
    if not constraints:
        return []

    if len(constraints) == 1:
        return constraints

    logger.info(f"Deduplicating {len(constraints)} constraints")

    unique_constraints = []
    seen_indices = set()

    for i, constraint in enumerate(constraints):
        if i in seen_indices:
            continue

        # Add this constraint
        unique_constraints.append(constraint)
        seen_indices.add(i)

        # Check for similar constraints
        for j in range(i + 1, len(constraints)):
            if j in seen_indices:
                continue

            similarity = constraint_semantic_similarity(
                constraint,
                constraints[j]
            )

            if similarity >= similarity_threshold:
                logger.debug(
                    f"Merging similar constraints (similarity: {similarity:.3f})"
                )
                # Mark as seen (will be merged)
                seen_indices.add(j)

                # Merge the constraints
                merged = merge_similar_constraints(
                    constraint,
                    constraints[j]
                )
                # Update the unique constraint with merged version
                unique_constraints[-1] = merged

    logger.info(f"Deduplicated to {len(unique_constraints)} unique constraints")
    return unique_constraints


def merge_similar_constraints(
    constraint1: Dict[str, Any],
    constraint2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two similar constraints into one.

    The merged constraint will:
    - Use the nl_text from the higher priority constraint
    - Combine predicates if both exist
    - Take the maximum priority
    - Take the maximum confidence

    Args:
        constraint1: First constraint dictionary
        constraint2: Second constraint dictionary

    Returns:
        Merged constraint dictionary
    """
    # Use constraint with higher priority as base
    priority1 = constraint1.get("priority", 0)
    priority2 = constraint2.get("priority", 0)

    if priority1 >= priority2:
        base = constraint1.copy()
        other = constraint2
    else:
        base = constraint2.copy()
        other = constraint1

    # Update priority and confidence to maximum
    base["priority"] = max(priority1, priority2)
    base["confidence"] = max(
        constraint1.get("confidence", 0.0),
        constraint2.get("confidence", 0.0)
    )

    # Combine source information
    source1 = constraint1.get("source", {})
    source2 = constraint2.get("source", {})

    if isinstance(source1, dict) and isinstance(source2, dict):
        base["source"] = {
            **source1,
            **source2,
            "merged": True
        }

    return base


def sort_by_priority(
    constraints: List[Dict[str, Any]],
    reverse: bool = True
) -> List[Dict[str, Any]]:
    """
    Sort constraints by priority.

    Higher priority constraints are typically:
    - Higher confidence
    - More critical error types
    - Explicitly marked as high priority

    Args:
        constraints: List of constraint dictionaries
        reverse: If True, sort high to low (default)

    Returns:
        Sorted list of constraints
    """
    def get_priority_score(constraint: Dict[str, Any]) -> float:
        """
        Compute a priority score for sorting.

        Considers:
        - Explicit priority field
        - Confidence
        - Error type (if available)
        """
        score = 0.0

        # Explicit priority (weight: 100)
        priority = constraint.get("priority", 0)
        score += priority * 100

        # Confidence (weight: 10)
        confidence = constraint.get("confidence", 0.0)
        score += confidence * 10

        # Error type bonus
        error_type = constraint.get("type", "")
        type_weights = {
            "contradiction": 5.0,
            "factual_incorrect": 4.0,
            "constraint_violation": 3.0,
            "numeric_contradiction": 3.0,
            "missing_justification": 1.0
        }
        score += type_weights.get(error_type, 0.0)

        # Immutable constraints get higher priority
        if constraint.get("immutable", False):
            score += 50

        return score

    sorted_constraints = sorted(
        constraints,
        key=get_priority_score,
        reverse=reverse
    )

    return sorted_constraints


def compute_priority(
    error: Dict[str, Any],
    verify_threshold: float = 0.75
) -> int:
    """
    Compute priority score for a constraint based on the error it addresses.

    Args:
        error: Error dictionary
        verify_threshold: Threshold for high-priority classification

    Returns:
        Priority score (0-100)
    """
    confidence = error.get("confidence", 0.0)
    error_type = error.get("type", "")

    # Base priority from confidence
    if confidence >= verify_threshold + 0.10:
        priority = 80  # Very high confidence
    elif confidence >= verify_threshold:
        priority = 60  # High confidence
    else:
        priority = 40  # Medium confidence

    # Adjust based on error type
    if error_type in ["contradiction", "factual_incorrect"]:
        priority += 10
    elif error_type in ["constraint_violation"]:
        priority += 5

    # Cap at 100
    priority = min(100, priority)

    return priority
