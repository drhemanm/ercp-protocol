"""
Verify Operator - Reasoning Verification
Author: ERCP Protocol Implementation
License: Apache-2.0

Implements the Verify operator (V) from the ERCP protocol.
Orchestrates multiple validators to detect errors in reasoning.
"""

from typing import List, Dict, Any, Optional
import hashlib

from .base import BaseOperator
from ..validators.nli_validator import NLIValidator
from ..validators.rule_validator import RuleValidator


class VerifyOperator(BaseOperator):
    """
    Verify reasoning for errors and constraint violations.

    This operator:
    1. Accepts reasoning text and constraints
    2. Calls NLI validator for contradiction detection
    3. Calls rule validator for rule-based checks
    4. Optionally calls fact checker (if enabled)
    5. Aggregates and deduplicates errors
    6. Sorts by confidence (highest first)
    7. Returns list of error objects
    """

    def __init__(
        self,
        nli_threshold: float = 0.75,
        **kwargs
    ):
        """
        Initialize the Verify operator.

        Args:
            nli_threshold: Confidence threshold for NLI contradictions
            **kwargs: Additional arguments passed to BaseOperator
        """
        super().__init__(**kwargs)
        self.nli_threshold = nli_threshold

        # Initialize validators
        self.nli_validator = NLIValidator(
            confidence_threshold=nli_threshold,
            logger=self.logger
        )
        self.rule_validator = RuleValidator(logger=self.logger)

    def _deduplicate_errors(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate overlapping errors.

        Args:
            errors: List of error dictionaries

        Returns:
            Deduplicated list of errors
        """
        if not errors:
            return []

        # Use a hash of span + type to identify duplicates
        seen = set()
        unique_errors = []

        for error in errors:
            error_hash = hashlib.md5(
                f"{error['type']}_{error['span']}".encode()
            ).hexdigest()

            if error_hash not in seen:
                seen.add(error_hash)
                unique_errors.append(error)

        self.logger.debug(
            f"Deduplicated {len(errors)} -> {len(unique_errors)} errors"
        )

        return unique_errors

    def _sort_by_confidence(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort errors by confidence (highest first).

        Args:
            errors: List of error dictionaries

        Returns:
            Sorted list of errors
        """
        return sorted(
            errors,
            key=lambda e: e.get('confidence', 0.0),
            reverse=True
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def execute(
        self,
        reasoning_text: str,
        reasoning_id: Optional[str] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        run_fact_check: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the Verify operator.

        Args:
            reasoning_text: The reasoning text to verify
            reasoning_id: Optional reasoning ID
            constraints: Optional list of constraints to check
            run_fact_check: Whether to run fact checking (optional)
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
                - errors: List of detected errors
                - error_count: Total number of errors
                - has_errors: Boolean indicating if errors were found
        """
        if constraints is None:
            constraints = []

        self.logger.info(
            f"Verifying reasoning (constraints: {len(constraints)}, "
            f"fact_check: {run_fact_check})"
        )

        # Split text into sentences
        sentences = self._split_into_sentences(reasoning_text)

        all_errors = []

        # Run NLI validator
        try:
            nli_errors = self.nli_validator.validate(
                reasoning_text=reasoning_text,
                sentences=sentences,
                constraints=constraints
            )
            all_errors.extend(nli_errors)
            self.logger.info(f"NLI validation found {len(nli_errors)} errors")
        except Exception as e:
            self.logger.error(f"NLI validation failed: {str(e)}")

        # Run rule validator
        try:
            rule_errors = self.rule_validator.validate(
                reasoning_text=reasoning_text,
                sentences=sentences,
                constraints=constraints
            )
            all_errors.extend(rule_errors)
            self.logger.info(f"Rule validation found {len(rule_errors)} errors")
        except Exception as e:
            self.logger.error(f"Rule validation failed: {str(e)}")

        # Run fact checker if enabled
        if run_fact_check:
            # Fact checker is optional for MVP - can be implemented later
            self.logger.info("Fact checking requested but not yet implemented")
            pass

        # Deduplicate errors
        unique_errors = self._deduplicate_errors(all_errors)

        # Sort by confidence
        sorted_errors = self._sort_by_confidence(unique_errors)

        self.logger.info(
            f"Verification complete: {len(sorted_errors)} unique errors found"
        )

        return {
            "errors": sorted_errors,
            "error_count": len(sorted_errors),
            "has_errors": len(sorted_errors) > 0
        }
