"""
Predicate DSL - Domain Specific Language for Constraints
Author: ERCP Protocol Implementation
License: Apache-2.0

Defines predicate types and validation logic for ERCP constraints.
Predicates are machine-readable representations of constraints.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import re


class Predicate(ABC):
    """
    Abstract base class for all predicates.

    A predicate is a machine-readable constraint that can be evaluated
    against reasoning text.
    """

    def __init__(self, predicate_name: str, args: Dict[str, Any]):
        """
        Initialize a predicate.

        Args:
            predicate_name: Name of the predicate type
            args: Dictionary of predicate arguments
        """
        self.predicate_name = predicate_name
        self.args = args

    @abstractmethod
    def validate(self, reasoning_text: str) -> bool:
        """
        Evaluate this predicate against reasoning text.

        Args:
            reasoning_text: The reasoning text to validate

        Returns:
            True if reasoning satisfies the predicate, False otherwise
        """
        raise NotImplementedError(f"{self.predicate_name} must implement validate()")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize predicate to dictionary.

        Returns:
            Dictionary representation of the predicate
        """
        return {
            "predicate_name": self.predicate_name,
            "args": self.args
        }

    def to_json(self) -> str:
        """
        Serialize predicate to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Predicate':
        """
        Deserialize predicate from dictionary.

        Args:
            data: Dictionary containing predicate data

        Returns:
            Predicate instance
        """
        predicate_name = data.get("predicate_name")
        args = data.get("args", {})

        # Map predicate names to classes
        predicate_map = {
            "equal": Equal,
            "not_equal": NotEqual,
            "less_than": LessThan,
            "greater_than": GreaterThan,
            "no_contradiction": NoContradiction,
            "temporal_order": TemporalOrder,
            "has_justification": HasJustification
        }

        predicate_class = predicate_map.get(predicate_name.lower())
        if not predicate_class:
            raise ValueError(f"Unknown predicate type: {predicate_name}")

        return predicate_class(**args)

    @classmethod
    def from_json(cls, json_str: str) -> 'Predicate':
        """
        Deserialize predicate from JSON string.

        Args:
            json_str: JSON string

        Returns:
            Predicate instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class Equal(Predicate):
    """
    Predicate: value1 == value2

    Example: Equal(entity="water boiling point", value="100°C")
    """

    def __init__(self, entity: str, value: str, **kwargs):
        """
        Initialize Equal predicate.

        Args:
            entity: The entity to check
            value: The expected value
        """
        super().__init__("equal", {"entity": entity, "value": value})
        self.entity = entity
        self.value = value

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if reasoning states that entity equals value.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if entity equals value in the text
        """
        # Simple pattern matching
        pattern = f"{re.escape(self.entity)}.*{re.escape(self.value)}"
        return bool(re.search(pattern, reasoning_text, re.IGNORECASE))


class NotEqual(Predicate):
    """
    Predicate: value1 != value2

    Example: NotEqual(entity="water boiling point", value="50°C")
    """

    def __init__(self, entity: str, value: str, **kwargs):
        """
        Initialize NotEqual predicate.

        Args:
            entity: The entity to check
            value: The value that should NOT be present
        """
        super().__init__("not_equal", {"entity": entity, "value": value})
        self.entity = entity
        self.value = value

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if reasoning does NOT state that entity equals value.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if entity does not equal value in the text
        """
        pattern = f"{re.escape(self.entity)}.*{re.escape(self.value)}"
        return not bool(re.search(pattern, reasoning_text, re.IGNORECASE))


class LessThan(Predicate):
    """
    Predicate: value1 < value2

    Example: LessThan(entity="altitude boiling point", reference="sea level boiling point")
    """

    def __init__(self, entity: str, reference: str, **kwargs):
        """
        Initialize LessThan predicate.

        Args:
            entity: The entity that should be less
            reference: The reference entity
        """
        super().__init__("less_than", {"entity": entity, "reference": reference})
        self.entity = entity
        self.reference = reference

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if reasoning implies entity < reference.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if entity < reference is implied
        """
        # Look for comparative language
        patterns = [
            f"{re.escape(self.entity)}.*lower.*{re.escape(self.reference)}",
            f"{re.escape(self.entity)}.*less.*{re.escape(self.reference)}",
            f"{re.escape(self.entity)}.*smaller.*{re.escape(self.reference)}",
        ]

        for pattern in patterns:
            if re.search(pattern, reasoning_text, re.IGNORECASE):
                return True

        return False


class GreaterThan(Predicate):
    """
    Predicate: value1 > value2

    Example: GreaterThan(entity="sea level pressure", reference="mountain pressure")
    """

    def __init__(self, entity: str, reference: str, **kwargs):
        """
        Initialize GreaterThan predicate.

        Args:
            entity: The entity that should be greater
            reference: The reference entity
        """
        super().__init__("greater_than", {"entity": entity, "reference": reference})
        self.entity = entity
        self.reference = reference

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if reasoning implies entity > reference.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if entity > reference is implied
        """
        patterns = [
            f"{re.escape(self.entity)}.*higher.*{re.escape(self.reference)}",
            f"{re.escape(self.entity)}.*greater.*{re.escape(self.reference)}",
            f"{re.escape(self.entity)}.*more.*{re.escape(self.reference)}",
        ]

        for pattern in patterns:
            if re.search(pattern, reasoning_text, re.IGNORECASE):
                return True

        return False


class NoContradiction(Predicate):
    """
    Predicate: No contradictions between statements

    Example: NoContradiction(topic="water boiling temperature")
    """

    def __init__(self, topic: Optional[str] = None, **kwargs):
        """
        Initialize NoContradiction predicate.

        Args:
            topic: Optional topic to focus on
        """
        super().__init__("no_contradiction", {"topic": topic})
        self.topic = topic

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if reasoning contains no contradictions.

        Note: This requires more sophisticated validation (NLI model).
        For now, returns True as a placeholder.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if no contradictions detected
        """
        # This should be validated by the NLI validator
        # Returning True here as a placeholder
        return True


class TemporalOrder(Predicate):
    """
    Predicate: Event A occurs before Event B

    Example: TemporalOrder(before="heating", after="boiling")
    """

    def __init__(self, before: str, after: str, **kwargs):
        """
        Initialize TemporalOrder predicate.

        Args:
            before: Event that should occur first
            after: Event that should occur second
        """
        super().__init__("temporal_order", {"before": before, "after": after})
        self.before = before
        self.after = after

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if reasoning respects temporal ordering.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if temporal order is correct
        """
        # Simple check: before event should appear before after event
        before_pos = reasoning_text.lower().find(self.before.lower())
        after_pos = reasoning_text.lower().find(self.after.lower())

        if before_pos == -1 or after_pos == -1:
            return True  # Can't validate if events not mentioned

        return before_pos < after_pos


class HasJustification(Predicate):
    """
    Predicate: Claim has supporting evidence

    Example: HasJustification(claim="water boils at 100°C")
    """

    def __init__(self, claim: str, **kwargs):
        """
        Initialize HasJustification predicate.

        Args:
            claim: The claim that should be justified
        """
        super().__init__("has_justification", {"claim": claim})
        self.claim = claim

    def validate(self, reasoning_text: str) -> bool:
        """
        Check if claim has justification in reasoning.

        Args:
            reasoning_text: The reasoning text

        Returns:
            True if claim appears with justification
        """
        # Look for claim with justification keywords nearby
        claim_pattern = re.escape(self.claim)
        justification_keywords = [
            "because", "since", "due to", "as", "given that",
            "based on", "according to", "evidence shows"
        ]

        # Find claim in text
        claim_match = re.search(claim_pattern, reasoning_text, re.IGNORECASE)
        if not claim_match:
            return True  # Claim not present, can't validate

        # Check for justification keywords within 100 characters
        start = max(0, claim_match.start() - 100)
        end = min(len(reasoning_text), claim_match.end() + 100)
        context = reasoning_text[start:end].lower()

        for keyword in justification_keywords:
            if keyword in context:
                return True

        return False
