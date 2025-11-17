"""
Base Operator Abstract Class
Author: ERCP Protocol Implementation
License: Apache-2.0

Defines the abstract base class for all ERCP operators.
All operators (Generate, Verify, Extract, Stabilize, Mutate) must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time
import uuid


class BaseOperator(ABC):
    """
    Abstract base class for all ERCP operators.

    Each operator must implement the execute() method and follow consistent
    patterns for logging, error handling, and result formatting.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the operator with a logger.

        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or self._create_default_logger()
        self.operator_name = self.__class__.__name__

    def _create_default_logger(self) -> logging.Logger:
        """Create a default logger for this operator."""
        logger = logging.getLogger(f"ercp.operators.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the operator's main logic.

        This method must be implemented by all subclasses.

        Args:
            **kwargs: Operator-specific parameters

        Returns:
            Dict containing the operator's output

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.operator_name} must implement execute()")

    def _log_start(self, trace_id: Optional[str] = None, **kwargs):
        """
        Log the start of operator execution.

        Args:
            trace_id: Optional trace ID for tracking
            **kwargs: Additional context to log
        """
        context = {
            "operator": self.operator_name,
            "trace_id": trace_id,
            **kwargs
        }
        self.logger.info(f"{self.operator_name}.started", extra=context)

    def _log_complete(self, trace_id: Optional[str] = None, duration: Optional[float] = None, **kwargs):
        """
        Log the completion of operator execution.

        Args:
            trace_id: Optional trace ID for tracking
            duration: Execution duration in seconds
            **kwargs: Additional context to log
        """
        context = {
            "operator": self.operator_name,
            "trace_id": trace_id,
            "duration_seconds": duration,
            **kwargs
        }
        self.logger.info(f"{self.operator_name}.completed", extra=context)

    def _log_error(self, error: Exception, trace_id: Optional[str] = None, **kwargs):
        """
        Log an error during operator execution.

        Args:
            error: The exception that occurred
            trace_id: Optional trace ID for tracking
            **kwargs: Additional context to log
        """
        context = {
            "operator": self.operator_name,
            "trace_id": trace_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        }
        self.logger.error(f"{self.operator_name}.failed", extra=context, exc_info=True)

    def run(self, trace_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the operator with logging and error handling.

        This method wraps the execute() method with consistent logging
        and error handling patterns.

        Args:
            trace_id: Optional trace ID for tracking
            **kwargs: Operator-specific parameters

        Returns:
            Dict containing the operator's output with metadata

        Raises:
            Exception: Re-raises any exception after logging
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())

        self._log_start(trace_id=trace_id, execution_id=execution_id)

        try:
            result = self.execute(**kwargs)

            duration = time.time() - start_time
            self._log_complete(
                trace_id=trace_id,
                execution_id=execution_id,
                duration=duration
            )

            # Add metadata to result
            if isinstance(result, dict):
                result["_metadata"] = {
                    "operator": self.operator_name,
                    "execution_id": execution_id,
                    "duration_seconds": duration,
                    "timestamp": time.time()
                }

            return result

        except Exception as e:
            self._log_error(e, trace_id=trace_id, execution_id=execution_id)
            raise

    def validate_input(self, required_fields: list, kwargs: dict) -> None:
        """
        Validate that required input fields are present.

        Args:
            required_fields: List of required field names
            kwargs: Input parameters to validate

        Raises:
            ValueError: If any required field is missing
        """
        missing = [field for field in required_fields if field not in kwargs]
        if missing:
            raise ValueError(
                f"{self.operator_name} missing required fields: {', '.join(missing)}"
            )
