"""
Extract Operator - Constraint Extraction from Errors
Author: ERCP Protocol Implementation
License: Apache-2.0

Implements the Extract operator (X) from the ERCP protocol.
Extracts constraints from detected errors using LLM meta-prompting.
"""

import uuid
from typing import List, Dict, Any, Optional
import re

from .base import BaseOperator
from ..models.model_registry import get_model_registry
from ..utils.constraint_utils import (
    deduplicate_constraints,
    sort_by_priority,
    compute_priority
)
from ..predicates.predicate_dsl import Predicate, NoContradiction


class ExtractOperator(BaseOperator):
    """
    Extract constraints from detected errors.

    This operator:
    1. Accepts list of errors and reasoning text
    2. For each error, generates constraint via meta-prompt to LLM
    3. Parses LLM response into nl_text
    4. Synthesizes predicate from nl_text
    5. Computes priority based on error confidence and type
    6. Deduplicates constraints
    7. Separates into constraints and candidate_constraints based on thresholds
    8. Returns both lists
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        **kwargs
    ):
        """
        Initialize the Extract operator.

        Args:
            model_name: Name of the LLM model to use for constraint generation
            **kwargs: Additional arguments passed to BaseOperator
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.registry = get_model_registry()

    def _build_meta_prompt(
        self,
        error: Dict[str, Any],
        reasoning_text: str
    ) -> str:
        """
        Build a meta-prompt to generate a constraint from an error.

        Args:
            error: Error dictionary
            reasoning_text: The reasoning text that contains the error

        Returns:
            Meta-prompt string
        """
        error_type = error.get("type", "unknown")
        excerpt = error.get("excerpt", "")
        evidence = error.get("evidence", {})

        prompt = f"""Given the following error in reasoning, generate a clear, specific constraint that would prevent this error in the future.

Error Type: {error_type}
Error Excerpt: {excerpt}
Evidence: {str(evidence)}

Generate a single, concise constraint in natural language that addresses this error.
The constraint should be specific, actionable, and help prevent similar errors.

Constraint:"""

        return prompt

    def _generate_constraint_text(
        self,
        error: Dict[str, Any],
        reasoning_text: str
    ) -> str:
        """
        Generate constraint natural language text using LLM.

        Args:
            error: Error dictionary
            reasoning_text: The reasoning text

        Returns:
            Generated constraint text
        """
        try:
            # Build meta-prompt
            prompt = self._build_meta_prompt(error, reasoning_text)

            # Get model
            model_info = self.registry.get_generate_model(self.model_name)
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            device = model_info["device"]

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            # Generate
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract constraint text (after "Constraint:")
            if "Constraint:" in generated_text:
                constraint_text = generated_text.split("Constraint:")[-1].strip()
            else:
                constraint_text = generated_text[len(prompt):].strip()

            # Clean up (take first sentence)
            constraint_text = constraint_text.split('.')[0].strip() + '.'

            self.logger.debug(f"Generated constraint: {constraint_text}")

            return constraint_text

        except Exception as e:
            self.logger.error(f"Constraint generation failed: {str(e)}")
            # Fallback: create simple constraint from error type
            return self._fallback_constraint_text(error)

    def _fallback_constraint_text(self, error: Dict[str, Any]) -> str:
        """
        Generate fallback constraint text when LLM generation fails.

        Args:
            error: Error dictionary

        Returns:
            Fallback constraint text
        """
        error_type = error.get("type", "unknown")

        fallback_templates = {
            "contradiction": "Avoid contradictory statements about the same topic",
            "factual_incorrect": "Ensure factual accuracy of all claims",
            "constraint_violation": "Respect all specified constraints",
            "numeric_contradiction": "Use consistent numerical values for the same entity",
            "unit_inconsistency": "Use consistent units of measurement",
            "missing_justification": "Provide justification for all claims"
        }

        return fallback_templates.get(
            error_type,
            "Ensure logical consistency in reasoning"
        )

    def _synthesize_predicate(
        self,
        nl_text: str,
        error: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesize a machine-readable predicate from natural language text.

        This is a simplified implementation. In production, this could use:
        - Semantic parsing
        - Rule-based pattern matching
        - LLM-based predicate generation

        Args:
            nl_text: Natural language constraint text
            error: Original error dictionary

        Returns:
            Predicate dictionary or None
        """
        error_type = error.get("type", "")

        # Simple rule-based predicate synthesis
        if error_type in ["contradiction", "constraint_violation"]:
            # Use NoContradiction predicate
            predicate = NoContradiction()
            return predicate.to_dict()

        # For other types, return None (no specific predicate)
        return None

    def _create_constraint(
        self,
        error: Dict[str, Any],
        nl_text: str,
        predicate: Optional[Dict[str, Any]],
        verify_threshold: float
    ) -> Dict[str, Any]:
        """
        Create a constraint dictionary from error and generated text.

        Args:
            error: Original error dictionary
            nl_text: Generated constraint natural language text
            predicate: Synthesized predicate (if any)
            verify_threshold: Threshold for determining priority

        Returns:
            Constraint dictionary
        """
        constraint_id = str(uuid.uuid4())
        priority = compute_priority(error, verify_threshold)

        constraint = {
            "constraint_id": constraint_id,
            "type": error.get("type", "unknown"),
            "priority": priority,
            "nl_text": nl_text,
            "predicate": predicate,
            "source": {
                "error_id": error.get("error_id"),
                "detected_by": error.get("detected_by"),
                "error_type": error.get("type")
            },
            "confidence": error.get("confidence", 0.0),
            "immutable": False
        }

        return constraint

    def execute(
        self,
        errors: List[Dict[str, Any]],
        reasoning_text: str,
        verify_threshold: float = 0.75,
        candidate_threshold: float = 0.60,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the Extract operator.

        Args:
            errors: List of detected errors
            reasoning_text: The reasoning text that contains errors
            verify_threshold: Confidence threshold for full constraints
            candidate_threshold: Confidence threshold for candidate constraints
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
                - constraints: High-confidence constraints
                - candidate_constraints: Medium-confidence constraints
                - total_extracted: Total number of constraints extracted
        """
        if not errors:
            self.logger.info("No errors to extract constraints from")
            return {
                "constraints": [],
                "candidate_constraints": [],
                "total_extracted": 0
            }

        self.logger.info(f"Extracting constraints from {len(errors)} errors")

        all_constraints = []

        for error in errors:
            try:
                # Generate constraint natural language text
                nl_text = self._generate_constraint_text(error, reasoning_text)

                # Synthesize predicate
                predicate = self._synthesize_predicate(nl_text, error)

                # Create constraint
                constraint = self._create_constraint(
                    error,
                    nl_text,
                    predicate,
                    verify_threshold
                )

                all_constraints.append(constraint)

                self.logger.debug(
                    f"Extracted constraint: {nl_text} "
                    f"(priority: {constraint['priority']}, "
                    f"confidence: {constraint['confidence']:.3f})"
                )

            except Exception as e:
                self.logger.error(f"Failed to extract constraint from error: {str(e)}")
                continue

        # Deduplicate constraints
        unique_constraints = deduplicate_constraints(all_constraints)

        # Sort by priority
        sorted_constraints = sort_by_priority(unique_constraints)

        # Separate into constraints and candidate_constraints
        constraints = [
            c for c in sorted_constraints
            if c.get("confidence", 0.0) >= verify_threshold
        ]

        candidate_constraints = [
            c for c in sorted_constraints
            if candidate_threshold <= c.get("confidence", 0.0) < verify_threshold
        ]

        self.logger.info(
            f"Extracted {len(constraints)} constraints and "
            f"{len(candidate_constraints)} candidate constraints"
        )

        return {
            "constraints": constraints,
            "candidate_constraints": candidate_constraints,
            "total_extracted": len(unique_constraints)
        }
