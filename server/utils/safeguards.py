"""
Runtime safeguards for ERCP execution.
Includes circuit breakers, timeouts, and resource limits.
"""

import asyncio
import time
import psutil
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager
from server.logging import logger


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Fast-fail mode after too many failures
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("circuit_breaker.half_open", message="Attempting recovery")
            else:
                raise Exception("Circuit breaker is OPEN - too many recent failures")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("circuit_breaker.closed", message="Circuit recovered")
        self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                "circuit_breaker.open",
                message="Circuit opened due to failures",
                failure_count=self.failure_count
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout


@asynccontextmanager
async def timeout_guard(seconds: float, operation_name: str = "operation"):
    """
    Async context manager for timeout enforcement.

    Usage:
        async with timeout_guard(30, "model_generation"):
            result = await long_running_operation()

    Args:
        seconds: Timeout in seconds
        operation_name: Name for logging

    Raises:
        asyncio.TimeoutError: If operation exceeds timeout
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.error(
            "timeout_guard.exceeded",
            operation=operation_name,
            timeout_seconds=seconds
        )
        raise asyncio.TimeoutError(f"{operation_name} exceeded timeout of {seconds}s")


class ResourceMonitor:
    """
    Monitor system resources and enforce limits.
    """

    def __init__(
        self,
        max_memory_percent: float = 80.0,
        max_cpu_percent: float = 90.0
    ):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent

    def check_resources(self) -> dict:
        """
        Check current resource usage.

        Returns:
            Dictionary with memory and CPU usage
        """
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)

        return {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "cpu_percent": cpu,
            "safe": memory.percent < self.max_memory_percent and cpu < self.max_cpu_percent
        }

    def enforce_limits(self) -> None:
        """
        Enforce resource limits, raise exception if exceeded.

        Raises:
            RuntimeError: If resource limits are exceeded
        """
        resources = self.check_resources()

        if resources["memory_percent"] > self.max_memory_percent:
            logger.error(
                "resource_monitor.memory_exceeded",
                memory_percent=resources["memory_percent"],
                limit=self.max_memory_percent
            )
            raise RuntimeError(
                f"Memory usage {resources['memory_percent']:.1f}% exceeds limit {self.max_memory_percent}%"
            )

        if resources["cpu_percent"] > self.max_cpu_percent:
            logger.warning(
                "resource_monitor.cpu_high",
                cpu_percent=resources["cpu_percent"],
                limit=self.max_cpu_percent
            )


class IterationGuard:
    """
    Guard against runaway iteration loops.
    """

    def __init__(self, max_iterations: int, max_duration_seconds: float = 600.0):
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds
        self.start_time = time.time()
        self.iteration_count = 0

    def check(self, current_iteration: int) -> None:
        """
        Check if iteration is within bounds.

        Args:
            current_iteration: Current iteration number

        Raises:
            RuntimeError: If limits are exceeded
        """
        self.iteration_count = current_iteration + 1
        elapsed = time.time() - self.start_time

        # Check iteration limit
        if self.iteration_count > self.max_iterations:
            logger.error(
                "iteration_guard.max_iterations",
                current=self.iteration_count,
                max=self.max_iterations
            )
            raise RuntimeError(
                f"Iteration count {self.iteration_count} exceeds maximum {self.max_iterations}"
            )

        # Check time limit
        if elapsed > self.max_duration_seconds:
            logger.error(
                "iteration_guard.max_duration",
                elapsed_seconds=elapsed,
                max_seconds=self.max_duration_seconds
            )
            raise RuntimeError(
                f"Execution time {elapsed:.1f}s exceeds maximum {self.max_duration_seconds}s"
            )
