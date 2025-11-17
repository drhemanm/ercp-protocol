"""
ERCP Predicates Package
Contains predicate types and DSL for constraint representation.
"""

from .predicate_dsl import (
    Predicate,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    NoContradiction,
    TemporalOrder,
    HasJustification
)

__all__ = [
    "Predicate",
    "Equal",
    "NotEqual",
    "LessThan",
    "GreaterThan",
    "NoContradiction",
    "TemporalOrder",
    "HasJustification"
]
