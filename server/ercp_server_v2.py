"""
ERCP Reference Server - Production-Ready Implementation
Integrates real ML operators, database, security, and monitoring.

Author: Dr. Heman Mohabeer — EvoLogics AI Lab
License: Apache-2.0
Version: 2.0 (Production)
"""

import time
import uuid
import os
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Import operators
from server.operators import (
    GenerateOperator,
    VerifyOperator,
    ExtractConstraintsOperator,
    StabilizeOperator,
    MutateOperator,
)

# Import database
from server.db import get_db, init_db, Trace, TraceEvent, Constraint, Error

# Import security and middleware
from server.middleware import (
    limiter,
    custom_rate_limit_handler,
    RateLimitExceeded,
    SanitizationMiddleware,
    add_cors_middleware,
)

# Import logging and metrics
from server.logging import logger, setup_logging
from server.metrics import metrics_endpoint, record_ercp_run, active_ercp_runs

# Import safeguards
from server.utils import ResourceMonitor, IterationGuard

# Setup logging
setup_logging()

# ============================
# Application Lifespan
# ============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("ercp.startup", message="ERCP Server starting up")

    # Initialize database
    try:
        await init_db()
        logger.info("ercp.startup", message="Database initialized")
    except Exception as e:
        logger.error("ercp.startup", message="Failed to initialize database", error=str(e))

    # Load ML models (done lazily in model_registry)
    logger.info("ercp.startup", message="ML models will be loaded on first use")

    yield

    # Shutdown
    logger.info("ercp.shutdown", message="ERCP Server shutting down")


# ============================
# FastAPI App
# ============================

app = FastAPI(
    title="ERCP Protocol Server (Production)",
    version="2.0",
    description="Production-ready ERCP protocol server with real ML components",
    lifespan=lifespan,
)

# Add middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
app.add_middleware(SanitizationMiddleware)
add_cors_middleware(app)

# ============================
# Pydantic Models
# ============================


class Problem(BaseModel):
    id: Optional[str] = None
    description: str
    metadata: Optional[Dict[str, Any]] = {}


class Config(BaseModel):
    model: str = Field(default="gpt2")
    max_iterations: int = Field(default=20, ge=1, le=100)
    max_constraints: int = Field(default=30, ge=1, le=100)
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    deterministic: bool = True
    verify_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    candidate_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=10, le=2000)


class RunRequest(BaseModel):
    trace_id: Optional[str] = None
    problem: Problem
    config: Config = Config()


# ============================
# Health & Metrics
# ============================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return await metrics_endpoint()


# ============================
# Main ERCP Endpoint
# ============================


@app.post("/ercp/v1/run")
@limiter.limit("10/minute")
async def run_ercp(
    request: Request, req: RunRequest, db: AsyncSession = Depends(get_db)
):
    """
    Run full ERCP protocol with real ML operators.

    This is the production implementation that:
    - Uses real ML models for all operators
    - Persists all data to database
    - Records metrics
    - Handles errors gracefully
    """
    start_time = time.time()
    trace_id = req.trace_id or str(uuid.uuid4())

    logger.info(
        "ercp.run.start",
        trace_id=trace_id,
        problem_id=req.problem.id,
        max_iterations=req.config.max_iterations,
    )

    # Track active runs
    active_ercp_runs.inc()

    try:
        # Create trace in database
        trace = Trace(
            trace_id=trace_id,
            problem_id=req.problem.id or trace_id,
            problem_description=req.problem.description,
            status="running",
            config=req.config.dict(),
        )
        db.add(trace)
        await db.commit()

        # Initialize operators
        generator = GenerateOperator(config=req.config.dict())
        verifier = VerifyOperator()
        extractor = ExtractConstraintsOperator()
        stabilizer = StabilizeOperator()

        # Initialize safeguards
        resource_monitor = ResourceMonitor(max_memory_percent=85.0, max_cpu_percent=95.0)
        iteration_guard = IterationGuard(
            max_iterations=req.config.max_iterations,
            max_duration_seconds=600.0  # 10 minutes max
        )

        # Run ERCP loop
        prev_reasoning = None
        constraints_accum = []
        iteration = 0

        for iteration in range(req.config.max_iterations):
            # Check iteration and resource limits
            iteration_guard.check(iteration)
            resource_monitor.enforce_limits()
            logger.info(
                "ercp.iteration.start",
                trace_id=trace_id,
                iteration=iteration,
                constraint_count=len(constraints_accum),
            )

            # ========== GENERATE ==========
            try:
                gen_start = time.time()
                gen_result = generator.execute(
                    problem=req.problem.description,
                    constraints=constraints_accum,
                    config=req.config.dict(),
                )
                reasoning_text = gen_result["reasoning_text"]
                gen_duration = time.time() - gen_start

                # Log event
                event = TraceEvent(
                    trace_id=trace_id,
                    operator="generate",
                    iteration=iteration,
                    output_summary={
                        "reasoning_id": gen_result["reasoning_id"],
                        "sentence_count": len(gen_result["sentences"]),
                    },
                    duration_ms=gen_duration * 1000,
                )
                db.add(event)

                logger.info(
                    "ercp.generate.complete",
                    trace_id=trace_id,
                    iteration=iteration,
                    duration_ms=gen_duration * 1000,
                )

            except Exception as e:
                logger.error(
                    "ercp.generate.error",
                    trace_id=trace_id,
                    iteration=iteration,
                    error=str(e),
                )
                raise

            # ========== VERIFY ==========
            try:
                verify_start = time.time()
                errors = verifier.execute(
                    reasoning_text=reasoning_text,
                    constraints=constraints_accum,
                    config=req.config.dict(),
                )
                verify_duration = time.time() - verify_start

                # Save errors to database
                for error in errors:
                    db.add(
                        Error(
                            trace_id=trace_id,
                            type=error["type"],
                            span=error.get("span"),
                            excerpt=error.get("excerpt"),
                            confidence=error["confidence"],
                            detected_by=error["detected_by"],
                            evidence=error.get("evidence"),
                        )
                    )

                # Log event
                event = TraceEvent(
                    trace_id=trace_id,
                    operator="verify",
                    iteration=iteration,
                    output_summary={"error_count": len(errors)},
                    duration_ms=verify_duration * 1000,
                )
                db.add(event)

                logger.info(
                    "ercp.verify.complete",
                    trace_id=trace_id,
                    iteration=iteration,
                    error_count=len(errors),
                    duration_ms=verify_duration * 1000,
                )

            except Exception as e:
                logger.error(
                    "ercp.verify.error",
                    trace_id=trace_id,
                    iteration=iteration,
                    error=str(e),
                )
                raise

            # ========== EXTRACT CONSTRAINTS ==========
            if errors:
                try:
                    extract_start = time.time()
                    extract_result = extractor.execute(
                        errors=errors,
                        reasoning_text=reasoning_text,
                        config=req.config.dict(),
                    )
                    new_constraints = extract_result["constraints"]
                    extract_duration = time.time() - extract_start

                    # Save constraints to database
                    for constraint in new_constraints:
                        db.add(
                            Constraint(
                                trace_id=trace_id,
                                type=constraint["type"],
                                priority=constraint["priority"],
                                nl_text=constraint["nl_text"],
                                predicate=constraint["predicate"],
                                source=constraint["source"],
                                confidence=constraint["confidence"],
                                immutable=constraint["immutable"],
                            )
                        )

                    constraints_accum.extend(new_constraints)

                    # Log event
                    event = TraceEvent(
                        trace_id=trace_id,
                        operator="extract",
                        iteration=iteration,
                        output_summary={"constraints_added": len(new_constraints)},
                        duration_ms=extract_duration * 1000,
                    )
                    db.add(event)

                    logger.info(
                        "ercp.extract.complete",
                        trace_id=trace_id,
                        iteration=iteration,
                        constraints_added=len(new_constraints),
                        duration_ms=extract_duration * 1000,
                    )

                except Exception as e:
                    logger.error(
                        "ercp.extract.error",
                        trace_id=trace_id,
                        iteration=iteration,
                        error=str(e),
                    )
                    raise

                # Check constraint cap
                if len(constraints_accum) >= req.config.max_constraints:
                    logger.warning(
                        "ercp.constraint_cap_reached",
                        trace_id=trace_id,
                        constraint_count=len(constraints_accum),
                    )
                    break

            # ========== STABILIZE ==========
            try:
                stab_start = time.time()
                stab_result = stabilizer.execute(
                    prev_reasoning=prev_reasoning,
                    curr_reasoning=reasoning_text,
                    threshold=req.config.similarity_threshold,
                    errors=errors,
                )
                stab_duration = time.time() - stab_start

                # Log event
                event = TraceEvent(
                    trace_id=trace_id,
                    operator="stabilize",
                    iteration=iteration,
                    output_summary={
                        "stable": stab_result["stable"],
                        "score": stab_result["score"],
                    },
                    duration_ms=stab_duration * 1000,
                )
                db.add(event)

                logger.info(
                    "ercp.stabilize.complete",
                    trace_id=trace_id,
                    iteration=iteration,
                    stable=stab_result["stable"],
                    score=stab_result["score"],
                    duration_ms=stab_duration * 1000,
                )

                # Check for convergence
                if stab_result["stable"]:
                    logger.info(
                        "ercp.converged", trace_id=trace_id, iteration=iteration
                    )
                    break

            except Exception as e:
                logger.error(
                    "ercp.stabilize.error",
                    trace_id=trace_id,
                    iteration=iteration,
                    error=str(e),
                )
                raise

            prev_reasoning = reasoning_text

        # Determine final status
        final_status = "converged" if stab_result.get("stable") else "partial"

        # Update trace
        trace.status = final_status
        trace.final_reasoning = gen_result
        trace.iteration_count = iteration + 1
        await db.commit()

        # Record metrics
        duration = time.time() - start_time
        record_ercp_run(
            status=final_status,
            iterations=iteration + 1,
            constraints=len(constraints_accum),
            duration=duration,
        )

        logger.info(
            "ercp.run.complete",
            trace_id=trace_id,
            status=final_status,
            iterations=iteration + 1,
            duration_seconds=duration,
        )

        # Return response
        return {
            "trace_id": trace_id,
            "status": final_status,
            "final_reasoning": gen_result,
            "constraints": constraints_accum,
            "iterations": iteration + 1,
            "duration_seconds": duration,
            "proto_version": "ercp-2.0",
        }

    except Exception as e:
        logger.error("ercp.run.error", trace_id=trace_id, error=str(e))

        # Update trace status
        if "trace" in locals():
            trace.status = "failed"
            await db.commit()

        raise HTTPException(status_code=500, detail=f"ERCP run failed: {str(e)}")

    finally:
        active_ercp_runs.dec()


# ============================
# Error Handlers
# ============================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "http.error", path=request.url.path, method=request.method, error=str(exc)
    )

    return JSONResponse(
        status_code=500, content={"error": "internal_server_error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.ercp_server_v2:app",
        host="0.0.0.0",
        port=8080,
        reload=os.getenv("ENVIRONMENT") == "development",
    )
