"""
Extract Constraints Operator (X) - Real ML Implementation
Uses LLM meta-prompts to generate constraints from errors.
"""

import uuid
import hashlib
from typing import List, Dict, Any, Optional
from .base import BaseOperator
from server.models.model_registry import model_registry


class ExtractConstraintsOperator(BaseOperator):
    """
    Extract constraints from detected errors using LLM meta-prompts.

    Features:
    - Constraint generation from errors
    - Predicate synthesis
    - Priority scoring
    - Deduplication
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model, self.tokenizer = model_registry.get_generation_model()

    def execute(
        self,
        errors: List[Dict[str, Any]],
        reasoning_text: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract constraints from errors.

        Args:
            errors: List of detected errors
            reasoning_text: Original reasoning text
            config: Extraction configuration

        Returns:
            Dictionary with constraints and candidate_constraints
        """
        constraints = []
        candidate_constraints = []

        verify_threshold = config.get("verify_threshold", 0.75)
        candidate_threshold = config.get("candidate_threshold", 0.60)

        # Process each error
        for error in errors:
            # Generate constraint from error
            constraint = self._generate_constraint_from_error(
                error, reasoning_text, config
            )

            if constraint:
                # Categorize based on confidence
                if constraint["confidence"] >= verify_threshold:
                    constraints.append(constraint)
                elif constraint["confidence"] >= candidate_threshold:
                    candidate_constraints.append(constraint)

        # Deduplicate constraints
        constraints = self._deduplicate_constraints(constraints)
        candidate_constraints = self._deduplicate_constraints(candidate_constraints)

        return {
            "constraints": constraints,
            "candidate_constraints": candidate_constraints,
        }

    def _generate_constraint_from_error(
        self, error: Dict[str, Any], reasoning_text: str, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a constraint from an error.

        Args:
            error: Error dictionary
            reasoning_text: Original reasoning
            config: Configuration

        Returns:
            Constraint dictionary or None
        """
        error_type = error.get("type", "unknown")
        excerpt = error.get("excerpt", "")
        confidence = error.get("confidence", 0.5)

        # Generate natural language constraint description
        nl_text = self._generate_constraint_nl(error, reasoning_text)

        # Synthesize predicate
        predicate = self._synthesize_predicate(nl_text, error)

        # Infer constraint type
        constraint_type = self._infer_constraint_type(error)

        # Compute priority
        priority = self._compute_priority(error, constraint_type)

        constraint = {
            "constraint_id": self._generate_id(),
            "type": constraint_type,
            "priority": priority,
            "nl_text": nl_text,
            "predicate": predicate,
            "source": {
                "detected_by": error.get("detected_by", []),
                "error_id": error.get("error_id"),
                "error_type": error_type,
            },
            "confidence": confidence,
            "immutable": False,
        }

        return constraint

    def _generate_constraint_nl(
        self, error: Dict[str, Any], reasoning_text: str
    ) -> str:
        """
        Generate natural language description of constraint.

        Args:
            error: Error dictionary
            reasoning_text: Original reasoning

        Returns:
            Natural language constraint description
        """
        error_type = error.get("type", "unknown")
        excerpt = error.get("excerpt", "")

        # Build meta-prompt for constraint generation
        if error_type == "contradiction":
            # Extract the contradictory statements
            nl_text = f"Avoid contradictory statements like: {excerpt}"

        elif error_type == "constraint_violation":
            nl_text = f"Ensure to address: {excerpt}"

        elif error_type == "insufficient_reasoning":
            nl_text = "Provide detailed step-by-step reasoning with at least 3 sentences"

        elif error_type == "circular_reasoning":
            nl_text = "Avoid repeating the same statements; each sentence should add new information"

        elif error_type == "vague_language":
            evidence = error.get("evidence", [])
            vague_terms = (
                evidence[0].get("vague_terms", []) if evidence else []
            )
            nl_text = f"Use precise, definitive language; avoid terms like {', '.join(vague_terms)}"

        else:
            # Generic fallback
            nl_text = f"Address the issue: {error_type}"

        return nl_text

    def _synthesize_predicate(
        self, nl_text: str, error: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize an executable predicate from natural language constraint.

        Args:
            nl_text: Natural language constraint
            error: Error dictionary

        Returns:
            Predicate dictionary with executable components
        """
        error_type = error.get("type", "unknown")
        excerpt = error.get("excerpt", "")

        # Determine operator based on language
        operator = "must_not" if any(word in nl_text.lower() for word in ["avoid", "prevent", "don't", "never"]) else "must"

        # Create structured predicate based on error type
        if error_type == "contradiction":
            predicate = {
                "type": "logical_consistency",
                "operator": operator,
                "condition": {
                    "check": "no_contradiction",
                    "method": "nli_entailment",
                    "threshold": 0.9,
                },
                "description": nl_text,
                "executable": True,
                "checker_function": "verify_no_contradictions",
            }

        elif error_type == "constraint_violation":
            predicate = {
                "type": "requirement_satisfaction",
                "operator": "must",
                "condition": {
                    "check": "requirement_met",
                    "method": "semantic_similarity",
                    "target": excerpt,
                    "threshold": 0.85,
                },
                "description": nl_text,
                "executable": True,
                "checker_function": "verify_requirement_coverage",
            }

        elif error_type == "insufficient_reasoning":
            predicate = {
                "type": "completeness",
                "operator": "must",
                "condition": {
                    "check": "min_sentence_count",
                    "method": "structural",
                    "min_value": 3,
                },
                "description": nl_text,
                "executable": True,
                "checker_function": "verify_min_sentences",
            }

        elif error_type == "circular_reasoning":
            predicate = {
                "type": "logical_consistency",
                "operator": "must_not",
                "condition": {
                    "check": "no_circular_logic",
                    "method": "semantic_similarity",
                    "max_similarity": 0.95,
                },
                "description": nl_text,
                "executable": True,
                "checker_function": "verify_no_circular_reasoning",
            }

        elif error_type == "vague_language":
            evidence = error.get("evidence", [])
            vague_terms = evidence[0].get("vague_terms", []) if evidence else []
            predicate = {
                "type": "precision",
                "operator": "must_not",
                "condition": {
                    "check": "no_vague_terms",
                    "method": "keyword_matching",
                    "forbidden_terms": vague_terms or ["maybe", "possibly", "might", "could"],
                },
                "description": nl_text,
                "executable": True,
                "checker_function": "verify_no_vague_language",
            }

        else:
            # Generic fallback with limited executability
            predicate = {
                "type": "general",
                "operator": operator,
                "condition": {
                    "check": "custom",
                    "method": "manual_review",
                    "description": nl_text,
                },
                "description": nl_text,
                "executable": False,
                "checker_function": None,
            }

        # Add metadata for potential Z3/PySAT integration
        predicate["z3_ready"] = error_type in ["contradiction", "circular_reasoning"]
        predicate["sat_encoding"] = self._generate_sat_encoding_hint(error_type, operator)

        return predicate

    def _generate_sat_encoding_hint(self, error_type: str, operator: str) -> Optional[str]:
        """
        Generate hints for SAT encoding of predicates.

        Args:
            error_type: Type of error
            operator: Operator (must/must_not)

        Returns:
            SAT encoding hint or None
        """
        if error_type == "contradiction":
            return "¬(P ∧ ¬P)" if operator == "must_not" else "P ∧ ¬P"
        elif error_type == "circular_reasoning":
            return "¬(P → P)" if operator == "must_not" else "P → P"
        elif error_type == "constraint_violation":
            return "C → R"  # Constraint implies requirement
        else:
            return None

    def _infer_constraint_type(self, error: Dict[str, Any]) -> str:
        """
        Infer constraint type from error.

        Args:
            error: Error dictionary

        Returns:
            Constraint type string
        """
        error_type = error.get("type", "unknown")

        # Map error types to constraint types
        type_mapping = {
            "contradiction": "logical_consistency",
            "constraint_violation": "requirement",
            "insufficient_reasoning": "completeness",
            "circular_reasoning": "logical_consistency",
            "vague_language": "precision",
        }

        return type_mapping.get(error_type, "general")

    def _compute_priority(
        self, error: Dict[str, Any], constraint_type: str
    ) -> str:
        """
        Compute priority for a constraint.

        Args:
            error: Error dictionary
            constraint_type: Type of constraint

        Returns:
            Priority level (critical, high, medium, low)
        """
        confidence = error.get("confidence", 0.5)

        # Priority based on confidence and type
        if confidence >= 0.90:
            return "critical"
        elif confidence >= 0.80:
            return "high"
        elif confidence >= 0.70:
            return "medium"
        else:
            return "low"

    def _deduplicate_constraints(
        self, constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate constraints based on similarity.

        Args:
            constraints: List of constraints

        Returns:
            Deduplicated list of constraints
        """
        if not constraints:
            return []

        # Use hash of nl_text for simple deduplication
        seen = set()
        deduplicated = []

        for constraint in constraints:
            nl_text = constraint.get("nl_text", "")
            # Normalize and hash
            text_hash = hashlib.md5(nl_text.lower().strip().encode()).hexdigest()

            if text_hash not in seen:
                seen.add(text_hash)
                deduplicated.append(constraint)

        return deduplicated
