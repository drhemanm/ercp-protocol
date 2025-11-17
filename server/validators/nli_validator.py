"""
NLI Validator - Natural Language Inference based Contradiction Detection
Author: ERCP Protocol Implementation
License: Apache-2.0

Uses a DeBERTa-MNLI model to detect contradictions between sentences
in the generated reasoning.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations

from ..models.model_registry import get_model_registry


class NLIValidator:
    """
    Validates reasoning using Natural Language Inference (NLI) models.

    This validator:
    1. Loads a DeBERTa-MNLI model (or similar NLI model)
    2. Splits reasoning text into sentences
    3. Checks all sentence pairs for contradictions
    4. Uses a confidence threshold to filter results
    5. Returns errors with type="contradiction", span, confidence, and evidence
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli",
        confidence_threshold: float = 0.75,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the NLI validator.

        Args:
            model_name: Name of the NLI model to use
            confidence_threshold: Minimum confidence for reporting contradictions
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.registry = get_model_registry()

    def _check_contradiction(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[bool, float, str]:
        """
        Check if hypothesis contradicts premise using NLI model.

        Args:
            premise: The premise sentence
            hypothesis: The hypothesis sentence

        Returns:
            Tuple of (is_contradiction, confidence, label)
        """
        try:
            nli_model = self.registry.get_nli_model(self.model_name)

            # Format input for NLI model
            input_text = f"{premise} [SEP] {hypothesis}"

            # Run inference
            result = nli_model(input_text)[0]

            label = result['label'].lower()
            score = result['score']

            # Different NLI models use different label formats
            # Common formats: "CONTRADICTION", "contradiction", "LABEL_2"
            is_contradiction = 'contradiction' in label or label == 'label_2'

            return is_contradiction, score, label

        except Exception as e:
            self.logger.error(f"NLI check failed: {str(e)}")
            return False, 0.0, "error"

    def _batch_check_contradictions(
        self,
        pairs: List[Tuple[int, int, str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Check multiple sentence pairs for contradictions in batch.

        Args:
            pairs: List of tuples (idx1, idx2, sent1, sent2)

        Returns:
            List of error dictionaries for detected contradictions
        """
        errors = []

        for idx1, idx2, sent1, sent2 in pairs:
            is_contradiction, confidence, label = self._check_contradiction(
                sent1, sent2
            )

            if is_contradiction and confidence >= self.confidence_threshold:
                error = {
                    "type": "contradiction",
                    "span": [idx1, idx2],
                    "excerpt": f"{sent1} <-> {sent2}",
                    "confidence": confidence,
                    "detected_by": "nli_validator",
                    "evidence": {
                        "nli_label": label,
                        "nli_score": confidence,
                        "sentence_1": sent1,
                        "sentence_2": sent2
                    }
                }
                errors.append(error)
                self.logger.info(
                    f"Contradiction detected: sentences {idx1}-{idx2} "
                    f"(confidence: {confidence:.3f})"
                )

        return errors

    def validate(
        self,
        reasoning_text: str,
        sentences: List[str],
        constraints: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate reasoning for contradictions.

        Args:
            reasoning_text: The full reasoning text
            sentences: List of sentences to check
            constraints: Optional list of constraints to validate against

        Returns:
            List of error dictionaries for detected contradictions
        """
        self.logger.info(
            f"Running NLI validation on {len(sentences)} sentences"
        )

        if len(sentences) < 2:
            self.logger.info("Less than 2 sentences, skipping NLI validation")
            return []

        errors = []

        # Check all pairs of sentences for contradictions
        pairs = []
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], start=i+1):
                pairs.append((i, j, sent1, sent2))

        # Batch check contradictions
        errors = self._batch_check_contradictions(pairs)

        # Also check sentences against constraints if provided
        if constraints:
            constraint_errors = self._validate_against_constraints(
                sentences, constraints
            )
            errors.extend(constraint_errors)

        self.logger.info(f"NLI validation found {len(errors)} contradictions")
        return errors

    def _validate_against_constraints(
        self,
        sentences: List[str],
        constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if sentences violate any constraints.

        Args:
            sentences: List of sentences
            constraints: List of constraint dictionaries

        Returns:
            List of error dictionaries for constraint violations
        """
        errors = []

        for constraint in constraints:
            nl_text = constraint.get("nl_text", "")
            if not nl_text:
                continue

            # Check each sentence against the constraint
            for i, sentence in enumerate(sentences):
                is_contradiction, confidence, label = self._check_contradiction(
                    nl_text, sentence
                )

                if is_contradiction and confidence >= self.confidence_threshold:
                    error = {
                        "type": "constraint_violation",
                        "span": [i, i],
                        "excerpt": sentence,
                        "confidence": confidence,
                        "detected_by": "nli_validator",
                        "evidence": {
                            "nli_label": label,
                            "nli_score": confidence,
                            "constraint": nl_text,
                            "sentence": sentence
                        }
                    }
                    errors.append(error)
                    self.logger.info(
                        f"Constraint violation detected: sentence {i} "
                        f"violates constraint '{nl_text[:50]}...'"
                    )

        return errors
