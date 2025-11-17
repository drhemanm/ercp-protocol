"""
Structured logging configuration for ERCP Protocol.
Uses structlog for production-ready JSON logging with request ID correlation.
"""

import os
import sys
import logging
import contextvars
from contextlib import contextmanager
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger

# Context variable for request-scoped logging context
_logging_context_var = contextvars.ContextVar("logging_context", default={})


def setup_logging():
    """
    Configure structured logging for the application.
    Includes support for request ID correlation and context propagation.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )

    # Custom processor to merge context from context variable
    def add_context_processor(logger, method_name, event_dict):
        """Add context from context variable to log event."""
        context = _logging_context_var.get({})
        event_dict.update(context)
        return event_dict

    if log_format == "json":
        # JSON format for production
        structlog.configure(
            processors=[
                add_context_processor,  # Add request context
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Console format for development
        structlog.configure(
            processors=[
                add_context_processor,  # Add request context
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


@contextmanager
def logging_context(**context: Any):
    """
    Context manager to add context to all logs within the context.

    Usage:
        with logging_context(request_id="123", user_id="456"):
            logger.info("Processing request")  # Will include request_id and user_id

    Args:
        **context: Key-value pairs to add to logging context
    """
    # Get current context
    current_context = _logging_context_var.get({}).copy()

    # Merge with new context
    current_context.update(context)

    # Set new context
    token = _logging_context_var.set(current_context)

    try:
        yield
    finally:
        # Restore previous context
        _logging_context_var.reset(token)


def get_logger(name: str = None):
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger
    """
    return structlog.get_logger(name)


# Initialize logging on import
setup_logging()

# Export default logger
logger = get_logger("ercp")
