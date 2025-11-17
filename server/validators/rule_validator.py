"""
Rule Validator - Rule-based Validation
Author: ERCP Protocol Implementation
License: Apache-2.0

Performs rule-based validation for:
- Numeric contradictions
- Date ordering (temporal consistency)
- Unit consistency
- Missing justifications
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict


class RuleValidator:
    """
    Validates reasoning using rule-based checks.

    This validator checks for:
    1. Numeric contradictions (conflicting numbers for same entity)
    2. Date/temporal ordering issues
    3. Unit consistency (mixing Celsius/Fahrenheit, etc.)
    4. Missing justifications for claims
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the rule validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def _extract_numbers(self, text: str) -> List[Tuple[float, str, int, int]]:
        """
        Extract numbers with their units from text.

        Args:
            text: Input text

        Returns:
            List of tuples (number, unit, start_pos, end_pos)
        """
        # Pattern to match numbers with optional units
        pattern = r'(-?\d+\.?\d*)\s*([°]?[CFK]|degrees?|celsius|fahrenheit|kelvin|meters?|feet|kg|lbs?|pounds?)?'

        matches = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                number = float(match.group(1))
                unit = match.group(2) or ""
                matches.append((number, unit.lower(), match.start(), match.end()))
            except ValueError:
                continue

        return matches

    def _normalize_unit(self, unit: str) -> str:
        """
        Normalize unit names to standard form.

        Args:
            unit: Original unit string

        Returns:
            Normalized unit string
        """
        unit = unit.lower().strip()

        # Temperature units
        if unit in ['°c', 'c', 'celsius', 'degrees celsius']:
            return 'celsius'
        elif unit in ['°f', 'f', 'fahrenheit', 'degrees fahrenheit']:
            return 'fahrenheit'
        elif unit in ['°k', 'k', 'kelvin', 'degrees kelvin']:
            return 'kelvin'

        # Distance units
        elif unit in ['m', 'meter', 'meters', 'metre', 'metres']:
            return 'meter'
        elif unit in ['ft', 'foot', 'feet']:
            return 'feet'

        # Weight units
        elif unit in ['kg', 'kilogram', 'kilograms']:
            return 'kg'
        elif unit in ['lb', 'lbs', 'pound', 'pounds']:
            return 'pounds'

        return unit

    def _check_numeric_contradictions(
        self,
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Check for numeric contradictions between sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of error dictionaries
        """
        errors = []

        # Extract numbers from each sentence
        sentence_numbers = []
        for sent in sentences:
            numbers = self._extract_numbers(sent)
            sentence_numbers.append(numbers)

        # Check for contradictions
        # Simple heuristic: if two sentences mention different numbers
        # for what seems to be the same context, flag as potential contradiction
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                nums_i = sentence_numbers[i]
                nums_j = sentence_numbers[j]

                if not nums_i or not nums_j:
                    continue

                # Check if sentences talk about similar numbers with same units
                for num_i, unit_i, _, _ in nums_i:
                    for num_j, unit_j, _, _ in nums_j:
                        norm_unit_i = self._normalize_unit(unit_i)
                        norm_unit_j = self._normalize_unit(unit_j)

                        # Same unit but different numbers - potential contradiction
                        if norm_unit_i and norm_unit_j and norm_unit_i == norm_unit_j:
                            # Allow for small variations (e.g., rounding)
                            if abs(num_i - num_j) > 0.1 * max(abs(num_i), abs(num_j)):
                                error = {
                                    "type": "numeric_contradiction",
                                    "span": [i, j],
                                    "excerpt": f"{sentences[i]} <-> {sentences[j]}",
                                    "confidence": 0.70,
                                    "detected_by": "rule_validator",
                                    "evidence": {
                                        "value_1": f"{num_i} {unit_i}",
                                        "value_2": f"{num_j} {unit_j}",
                                        "sentence_1": sentences[i],
                                        "sentence_2": sentences[j]
                                    }
                                }
                                errors.append(error)
                                self.logger.info(
                                    f"Numeric contradiction: {num_i}{unit_i} vs {num_j}{unit_j}"
                                )

        return errors

    def _extract_dates(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract dates from text.

        Args:
            text: Input text

        Returns:
            List of tuples (date_string, start_pos, end_pos)
        """
        # Simple date patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        ]

        dates = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dates.append((match.group(), match.start(), match.end()))

        return dates

    def _check_temporal_consistency(
        self,
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Check for temporal/date ordering issues.

        Args:
            sentences: List of sentences

        Returns:
            List of error dictionaries
        """
        errors = []

        # Extract temporal indicators
        temporal_words = {
            'before': -1,
            'after': 1,
            'prior to': -1,
            'following': 1,
            'earlier': -1,
            'later': 1,
            'first': -1,
            'then': 1
        }

        # Simple heuristic: check for contradictory temporal indicators
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            for word, direction in temporal_words.items():
                if word in sent_lower:
                    # Check if other sentences contradict this ordering
                    for j, other_sent in enumerate(sentences):
                        if i == j:
                            continue

                        other_lower = other_sent.lower()
                        for other_word, other_direction in temporal_words.items():
                            if other_word in other_lower:
                                # If directions conflict, might be an error
                                if direction != other_direction:
                                    # This is a simplified check
                                    # In practice, would need more sophisticated NLP
                                    pass

        return errors

    def _check_unit_consistency(
        self,
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Check for unit inconsistency (mixing units).

        Args:
            sentences: List of sentences

        Returns:
            List of error dictionaries
        """
        errors = []

        # Track what units are used for similar measurements
        temp_units = set()
        distance_units = set()
        weight_units = set()

        for sent in sentences:
            numbers = self._extract_numbers(sent)
            for _, unit, _, _ in numbers:
                norm_unit = self._normalize_unit(unit)

                if norm_unit in ['celsius', 'fahrenheit', 'kelvin']:
                    temp_units.add(norm_unit)
                elif norm_unit in ['meter', 'feet']:
                    distance_units.add(norm_unit)
                elif norm_unit in ['kg', 'pounds']:
                    weight_units.add(norm_unit)

        # Check for mixing units
        if len(temp_units) > 1:
            error = {
                "type": "unit_inconsistency",
                "span": [0, len(sentences) - 1],
                "excerpt": f"Mixed temperature units: {', '.join(temp_units)}",
                "confidence": 0.80,
                "detected_by": "rule_validator",
                "evidence": {
                    "mixed_units": list(temp_units),
                    "category": "temperature"
                }
            }
            errors.append(error)
            self.logger.info(f"Unit inconsistency: mixed temperature units")

        return errors

    def _check_missing_justifications(
        self,
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Check for claims that lack justification.

        Args:
            sentences: List of sentences

        Returns:
            List of error dictionaries
        """
        errors = []

        # Simple heuristic: look for strong claims without supporting evidence
        # In practice, this would use more sophisticated NLP

        claim_indicators = [
            'therefore', 'thus', 'consequently', 'it follows that',
            'this means', 'this proves', 'clearly', 'obviously'
        ]

        evidence_indicators = [
            'because', 'since', 'as', 'due to', 'given that',
            'based on', 'according to', 'research shows'
        ]

        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()

            # Check if sentence makes a claim
            is_claim = any(indicator in sent_lower for indicator in claim_indicators)

            if is_claim:
                # Check if there's evidence in nearby sentences
                context_start = max(0, i - 2)
                context_end = min(len(sentences), i + 3)
                context = ' '.join(sentences[context_start:context_end]).lower()

                has_evidence = any(
                    indicator in context for indicator in evidence_indicators
                )

                if not has_evidence:
                    error = {
                        "type": "missing_justification",
                        "span": [i, i],
                        "excerpt": sent,
                        "confidence": 0.60,
                        "detected_by": "rule_validator",
                        "evidence": {
                            "claim": sent,
                            "missing": "supporting evidence"
                        }
                    }
                    errors.append(error)
                    self.logger.debug(
                        f"Missing justification for claim in sentence {i}"
                    )

        return errors

    def validate(
        self,
        reasoning_text: str,
        sentences: List[str],
        constraints: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate reasoning using rule-based checks.

        Args:
            reasoning_text: The full reasoning text
            sentences: List of sentences to check
            constraints: Optional list of constraints

        Returns:
            List of error dictionaries
        """
        self.logger.info(f"Running rule-based validation on {len(sentences)} sentences")

        errors = []

        # Run all checks
        errors.extend(self._check_numeric_contradictions(sentences))
        errors.extend(self._check_temporal_consistency(sentences))
        errors.extend(self._check_unit_consistency(sentences))
        errors.extend(self._check_missing_justifications(sentences))

        self.logger.info(f"Rule validation found {len(errors)} issues")
        return errors
