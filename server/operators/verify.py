"""
Verify Operator (V) - Real ML Implementation
Uses NLI models and rule-based validators for error detection.
"""

import uuid
from typing import List, Dict, Any, Optional
from .base import BaseOperator
from server.models.model_registry import model_registry


class VerifyOperator(BaseOperator):
    """
    Verify reasoning for errors using multiple detection methods.

    Features:
    - NLI-based contradiction detection
    - Constraint validation
    - Confidence scoring
    - Evidence collection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.nli_pipeline = model_registry.get_nli_pipeline()
        self.nlp = model_registry.get_spacy_nlp()

    def execute(
        self,
        reasoning_text: str,
        constraints: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Verify reasoning and detect errors.

        Args:
            reasoning_text: Generated reasoning text
            constraints: List of constraints to validate against
            config: Verification configuration

        Returns:
            List of error dictionaries
        """
        errors = []

        # Segment into sentences
        doc = self.nlp(reasoning_text)
        sentences = [sent.text.strip() for sent in doc.sents]

        # 1. Detect contradictions using NLI
        contradiction_errors = self._detect_contradictions(sentences, config)
        errors.extend(contradiction_errors)

        # 2. Validate against constraints
        constraint_errors = self._validate_constraints(
            reasoning_text, sentences, constraints, config
        )
        errors.extend(constraint_errors)

        # 3. Rule-based validation
        rule_errors = self._rule_based_validation(sentences, config)
        errors.extend(rule_errors)

        return errors

    def _detect_contradictions(
        self, sentences: List[str], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions between sentences using NLI.

        Args:
            sentences: List of sentences
            config: Configuration with nli_threshold

        Returns:
            List of contradiction errors
        """
        errors = []
        threshold = config.get("nli_threshold", 0.75)

        # Compare each pair of sentences
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i + 1 :], start=i + 1):
                # Skip very short sentences
                if len(sent1.split()) < 3 or len(sent2.split()) < 3:
                    continue

                # Run NLI model
                try:
                    result = self.nli_pipeline(f"{sent1} [SEP] {sent2}")[0]

                    if (
                        result["label"].upper() == "CONTRADICTION"
                        and result["score"] >= threshold
                    ):
                        errors.append(
                            {
                                "error_id": self._generate_id(),
                                "type": "contradiction",
                                "span": [i, j],
                                "excerpt": f"{sent1} <-> {sent2}",
                                "confidence": result["score"],
                                "detected_by": ["nli"],
                                "evidence": [
                                    {
                                        "source": "nli",
                                        "label": result["label"],
                                        "score": result["score"],
                                    }
                                ],
                            }
                        )
                except Exception as e:
                    self._log("error", f"NLI error: {str(e)}")

        return errors

    def _validate_constraints(
        self,
        reasoning_text: str,
        sentences: List[str],
        constraints: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Validate reasoning against constraints.

        Args:
            reasoning_text: Full reasoning text
            sentences: Segmented sentences
            constraints: Constraints to validate
            config: Configuration

        Returns:
            List of constraint violation errors
        """
        errors = []

        for constraint in constraints:
            constraint_type = constraint.get("type", "unknown")
            nl_text = constraint.get("nl_text", "")
            predicate = constraint.get("predicate", {})

            # Check if constraint is satisfied
            is_satisfied = self._check_constraint_satisfaction(
                reasoning_text, sentences, constraint, config
            )

            if not is_satisfied:
                errors.append(
                    {
                        "error_id": self._generate_id(),
                        "type": "constraint_violation",
                        "span": None,  # Could be improved with span detection
                        "excerpt": nl_text,
                        "confidence": constraint.get("confidence", 0.8),
                        "detected_by": ["constraint_validator"],
                        "evidence": [
                            {
                                "source": "constraint",
                                "constraint_id": constraint.get("constraint_id"),
                                "constraint_type": constraint_type,
                            }
                        ],
                    }
                )

        return errors

    def _check_constraint_satisfaction(
        self,
        reasoning_text: str,
        sentences: List[str],
        constraint: Dict[str, Any],
        config: Dict[str, Any],
    ) -> bool:
        """
        Check if a constraint is satisfied.

        Args:
            reasoning_text: Full reasoning text
            sentences: Segmented sentences
            constraint: Constraint dictionary
            config: Configuration

        Returns:
            True if satisfied, False otherwise
        """
        constraint_type = constraint.get("type", "unknown")
        nl_text = constraint.get("nl_text", "").lower()

        # Simple keyword-based validation (can be improved)
        # Extract key terms from constraint
        key_terms = self._extract_key_terms(nl_text)

        # Check if key terms appear in reasoning
        reasoning_lower = reasoning_text.lower()
        matches = sum(1 for term in key_terms if term in reasoning_lower)

        # Consider satisfied if at least 50% of key terms present
        satisfaction_ratio = matches / len(key_terms) if key_terms else 1.0

        return satisfaction_ratio >= 0.5

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from constraint text.

        Args:
            text: Constraint text

        Returns:
            List of key terms
        """
        # Parse with spaCy
        doc = self.nlp(text)

        # Extract nouns, verbs, and adjectives
        key_terms = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] and not token.is_stop:
                key_terms.append(token.lemma_.lower())

        return key_terms

    def _rule_based_validation(
        self, sentences: List[str], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply rule-based validators.

        Args:
            sentences: List of sentences
            config: Configuration

        Returns:
            List of errors from rule-based validation
        """
        errors = []

        # Rule 1: Check for empty or very short reasoning
        if len(sentences) < 2:
            errors.append(
                {
                    "error_id": self._generate_id(),
                    "type": "insufficient_reasoning",
                    "span": None,
                    "excerpt": "Reasoning is too brief",
                    "confidence": 0.95,
                    "detected_by": ["rule_validator"],
                    "evidence": [
                        {
                            "source": "rule",
                            "rule_name": "minimum_length",
                            "sentence_count": len(sentences),
                        }
                    ],
                }
            )

        # Rule 2: Check for circular reasoning (repeated sentences)
        sentence_set = set()
        for i, sent in enumerate(sentences):
            sent_normalized = sent.lower().strip()
            if sent_normalized in sentence_set:
                errors.append(
                    {
                        "error_id": self._generate_id(),
                        "type": "circular_reasoning",
                        "span": [i],
                        "excerpt": sent,
                        "confidence": 0.90,
                        "detected_by": ["rule_validator"],
                        "evidence": [
                            {"source": "rule", "rule_name": "no_repetition"}
                        ],
                    }
                )
            sentence_set.add(sent_normalized)

        # Rule 3: Check for vague language
        vague_terms = ["maybe", "possibly", "might", "could be", "perhaps"]
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            found_vague = [term for term in vague_terms if term in sent_lower]
            if found_vague:
                errors.append(
                    {
                        "error_id": self._generate_id(),
                        "type": "vague_language",
                        "span": [i],
                        "excerpt": sent,
                        "confidence": 0.70,
                        "detected_by": ["rule_validator"],
                        "evidence": [
                            {
                                "source": "rule",
                                "rule_name": "no_vague_language",
                                "vague_terms": found_vague,
                            }
                        ],
                    }
                )

        return errors
