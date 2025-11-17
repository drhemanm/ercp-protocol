"""
Prometheus metrics for ERCP Protocol.
Tracks performance and usage statistics.
"""

import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import Response

# Check if metrics are enabled
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# ============================================
# Define Metrics
# ============================================

# ERCP run metrics
ercp_runs_total = Counter(
    "ercp_runs_total",
    "Total number of ERCP runs",
    ["status"],  # converged, infeasible, partial, failed
)

ercp_iteration_count = Histogram(
    "ercp_iteration_count",
    "Number of iterations per ERCP run",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

ercp_constraint_count = Histogram(
    "ercp_constraint_count",
    "Number of constraints extracted per run",
    buckets=[0, 1, 5, 10, 20, 30, 50],
)

ercp_error_count = Histogram(
    "ercp_error_count",
    "Number of errors detected per verification",
    buckets=[0, 1, 2, 5, 10, 20, 50],
)

ercp_duration_seconds = Histogram(
    "ercp_duration_seconds",
    "ERCP run duration in seconds",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# Operator-specific metrics
operator_duration_seconds = Histogram(
    "ercp_operator_duration_seconds",
    "Operator execution duration in seconds",
    ["operator"],  # generate, verify, extract, stabilize, mutate
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

operator_calls_total = Counter(
    "ercp_operator_calls_total",
    "Total operator calls",
    ["operator", "status"],  # success, error
)

# Model metrics
model_inference_duration_seconds = Histogram(
    "ercp_model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name", "operator"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

model_token_count = Histogram(
    "ercp_model_token_count",
    "Number of tokens generated",
    ["model_name"],
    buckets=[10, 50, 100, 200, 500, 1000, 2000],
)

# API metrics
http_requests_total = Counter(
    "ercp_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "ercp_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5],
)

# Database metrics
db_query_duration_seconds = Histogram(
    "ercp_db_query_duration_seconds",
    "Database query duration in seconds",
    ["operation"],  # select, insert, update, delete
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1],
)

# Active gauges
active_ercp_runs = Gauge(
    "ercp_active_runs",
    "Number of currently running ERCP executions",
)

active_model_inferences = Gauge(
    "ercp_active_model_inferences",
    "Number of currently running model inferences",
)


# ============================================
# Metrics endpoint
# ============================================


async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Response with metrics in Prometheus format
    """
    if not ENABLE_METRICS:
        return Response(
            content="Metrics disabled", status_code=404, media_type="text/plain"
        )

    return Response(
        content=generate_latest(REGISTRY), media_type="text/plain; charset=utf-8"
    )


# ============================================
# Helper functions
# ============================================


def record_ercp_run(status: str, iterations: int, constraints: int, duration: float):
    """
    Record metrics for an ERCP run.

    Args:
        status: Run status
        iterations: Number of iterations
        constraints: Number of constraints
        duration: Duration in seconds
    """
    if not ENABLE_METRICS:
        return

    ercp_runs_total.labels(status=status).inc()
    ercp_iteration_count.observe(iterations)
    ercp_constraint_count.observe(constraints)
    ercp_duration_seconds.observe(duration)


def record_operator_call(operator: str, duration: float, status: str = "success"):
    """
    Record metrics for an operator call.

    Args:
        operator: Operator name
        duration: Duration in seconds
        status: Call status
    """
    if not ENABLE_METRICS:
        return

    operator_duration_seconds.labels(operator=operator).observe(duration)
    operator_calls_total.labels(operator=operator, status=status).inc()
