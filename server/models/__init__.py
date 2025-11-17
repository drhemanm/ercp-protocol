"""
Models package for ERCP Protocol
Handles ML model loading and management.
"""

from .model_registry import model_registry, ModelRegistry

__all__ = ["model_registry", "ModelRegistry"]
