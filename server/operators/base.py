"""
Base operator interface for ERCP protocol.
All operators inherit from BaseOperator.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import uuid
import time


class BaseOperator(ABC):
    """Abstract base class for all ERCP operators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize operator with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.operator_name = self.__class__.__name__

    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the operator.

        Must be implemented by subclasses.
        Returns a dictionary with operation results.
        """
        pass

    def _generate_id(self) -> str:
        """Generate a unique ID for tracking."""
        return str(uuid.uuid4())

    def _timestamp(self) -> float:
        """Get current timestamp."""
        return time.time()

    def _log(self, level: str, message: str, **kwargs):
        """Log operator activity (placeholder for structured logging)."""
        print(f"[{level.upper()}] {self.operator_name}: {message}", kwargs)
