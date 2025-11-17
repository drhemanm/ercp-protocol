# server/ercp_server.py
"""
ERCP Reference Server — FastAPI Implementation (Production v1.0)
Author: Dr. Heman Mohabeer — EvoLogics AI Lab
License: Apache-2.0

Production-ready ERCP server with:
- Real ML operators (Generate, Verify, Extract, Stabilize)
- Database persistence (PostgreSQL with async SQLAlchemy)
- Comprehensive logging
- Error handling
"""

import os
import uuid
import time
import hashlib
import hmac
import json
import logging
from typing import List, Optional, Any, Dict
from uuid import UUID
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

# Import operators
from .operators import GenerateOperator, VerifyOperator, ExtractOperator, StabilizeOperator
from .models.model_registry import get_model_registry

# Import database components
from .db.database import get_db, check_connection, close_db
from .db.repositories import (
    TraceRepository,
    TraceEventRepository,
    ConstraintRepository,
    ModelCacheRepository
)

# Import authentication and middleware
from .auth import optional_auth, get_api_key, optional_api_key
from .middleware import setup_cors, limiter, sanitize_input
from .middleware.logging_middleware import LoggingMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================
# CONFIGURATION
# ============================

APP_SECRET = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION").encode()
PROTO_VERSION = "ercp-1.0"

# Default configurations from environment
DEFAULT_MODEL = os.getenv("DEFAULT_GENERATE_MODEL", "gpt2")
DEFAULT_MAX_ITERATIONS = int(os.getenv("DEFAULT_MAX_ITERATIONS", "20"))
DEFAULT_MAX_CONSTRAINTS = int(os.getenv("DEFAULT_MAX_CONSTRAINTS", "30"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.95"))
DEFAULT_VERIFY_THRESHOLD = float(os.getenv("DEFAULT_VERIFY_THRESHOLD", "0.75"))
DEFAULT_CANDIDATE_THRESHOLD = float(os.getenv("DEFAULT_CANDIDATE_THRESHOLD", "0.60"))


# ============================
# Utilities
# ============================

def make_trace_id() -> UUID:
    """Generate a new trace ID."""
    return uuid.uuid4()


def sign_payload(payload: dict) -> str:
    """Compute HMAC signature of a payload."""
    raw = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(APP_SECRET, raw, hashlib.sha256).hexdigest()


def model_fingerprint(model_name: str = "local-llm") -> str:
    """
    Generate model fingerprint.
    In production, this should be the actual SHA256 of model weights.
    """
    return f"{model_name}-sha256-placeholder"


# ============================
# Pydantic Models (Schemas)
# ============================

class Problem(BaseModel):
    id: Optional[str] = None
    description: str
    metadata: Optional[dict] = {}


class Config(BaseModel):
    model: str = Field(default_factory=lambda: DEFAULT_MODEL)
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_constraints: int = DEFAULT_MAX_CONSTRAINTS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    temperature: float = 0.0
    deterministic: bool = True
    verify_threshold: float = DEFAULT_VERIFY_THRESHOLD
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD


class RunRequest(BaseModel):
    trace_id: Optional[str] = None
    problem: Problem
    config: Config = Field(default_factory=Config)


class GenerateRequest(BaseModel):
    trace_id: Optional[str] = None
    problem: Problem
    constraints: list = []
    gen_config: dict = {}


class VerifyRequest(BaseModel):
    trace_id: Optional[str] = None
    reasoning_id: str
    reasoning_text: str
    constraints: list = []
    verify_config: dict = {}


class ExtractRequest(BaseModel):
    trace_id: Optional[str] = None
    errors: list
    reasoning_text: str
    extract_config: dict = {}


class StabilizeRequest(BaseModel):
    trace_id: Optional[str] = None
    reasoning_prev: Optional[str] = None
    reasoning_curr: str
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    errors: list = []


class MutateRequest(BaseModel):
    trace_id: Optional[str] = None
    problem: Problem
    reasoning_text: str
    mutation_strategy: str = "decompose"
    mutation_config: dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    timestamp: float


class TraceResponse(BaseModel):
    trace_id: str
    problem_id: Optional[str]
    problem_description: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    config: Optional[dict]
    final_reasoning: Optional[str]
    iteration_count: int
    constraint_count: int
    error_count: int
    events: Optional[List[dict]] = None
    constraints: Optional[List[dict]] = None
    errors: Optional[List[dict]] = None


# ============================
# FastAPI App
# ============================

app = FastAPI(
    title="ERCP Protocol Reference Server",
    version="1.0",
    description="Production implementation for ERCP protocol (spec v1.0)"
)

# ============================
# Configure Middleware & Security
# ============================

# CORS - must be first
setup_cors(app)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# Input sanitization middleware
app.middleware("http")(sanitize_input)

logger.info("Security middleware configured successfully")


# ============================
# Initialize Operators (Singleton)
# ============================

generate_operator = None
verify_operator = None
extract_operator = None
stabilize_operator = None


def get_generate_operator() -> GenerateOperator:
    """Get or create Generate operator."""
    global generate_operator
    if generate_operator is None:
        generate_operator = GenerateOperator(model_name=DEFAULT_MODEL)
        logger.info("Initialized GenerateOperator")
    return generate_operator


def get_verify_operator() -> VerifyOperator:
    """Get or create Verify operator."""
    global verify_operator
    if verify_operator is None:
        verify_operator = VerifyOperator(nli_threshold=DEFAULT_VERIFY_THRESHOLD)
        logger.info("Initialized VerifyOperator")
    return verify_operator


def get_extract_operator() -> ExtractOperator:
    """Get or create Extract operator."""
    global extract_operator
    if extract_operator is None:
        extract_operator = ExtractOperator(model_name=DEFAULT_MODEL)
        logger.info("Initialized ExtractOperator")
    return extract_operator


def get_stabilize_operator() -> StabilizeOperator:
    """Get or create Stabilize operator."""
    global stabilize_operator
    if stabilize_operator is None:
        stabilize_operator = StabilizeOperator()
        logger.info("Initialized StabilizeOperator")
    return stabilize_operator


# ============================
# Lifecycle Events
# ============================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting ERCP server...")
    
    # Check database connection
    db_ok = await check_connection()
    if not db_ok:
        logger.warning("Database connection failed - server may not function properly")
    
    # Warm up models if configured
    warm_up_models = os.getenv("WARM_UP_MODELS", "").split(",")
    if warm_up_models and warm_up_models[0]:
        logger.info(f"Warming up models: {warm_up_models}")
        registry = get_model_registry()
        registry.warm_up(warm_up_models)
    
    logger.info("ERCP server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ERCP server...")
    await close_db()
    logger.info("ERCP server shut down successfully")


# ============================
# Health Check Endpoint
# ============================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_ok = await check_connection()
    
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=PROTO_VERSION,
        database="connected" if db_ok else "disconnected",
        timestamp=time.time()
    )


# ============================
# Main ERCP Run Endpoint
# ============================

@app.post("/ercp/v1/run")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def run_ercp(
    req: RunRequest,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(optional_api_key)
):
    """
    Main ERCP execution endpoint.

    Runs the complete ERCP loop: Generate → Verify → Extract → Stabilize
    Persists all trace data to database.

    Authentication: Optional (API key via X-API-Key header)
    Rate limit: 10 requests/minute per IP
    """
    trace_id = UUID(req.trace_id) if req.trace_id else make_trace_id()
    problem = req.problem
    config = req.config
    
    logger.info(f"Starting ERCP run: trace_id={trace_id}")
    
    try:
        # Create trace record
        trace = await TraceRepository.create_trace(
            db=db,
            trace_id=trace_id,
            problem_description=problem.description,
            problem_id=problem.id,
            config=config.dict(),
            status="running"
        )
        await db.commit()
        
        # Get operators
        gen_op = get_generate_operator()
        ver_op = get_verify_operator()
        ext_op = get_extract_operator()
        stab_op = get_stabilize_operator()
        
        # Iteration loop
        prev_reasoning = None
        constraints_accum = []
        iteration = 0
        status = "running"
        final_reasoning_obj = None
        
        for iteration in range(config.max_iterations):
            logger.info(f"Iteration {iteration} for trace {trace_id}")
            
            # ------------------------------
            # G — Generate
            # ------------------------------
            try:
                gen_result = gen_op.execute(
                    problem_description=problem.description,
                    constraints=constraints_accum,
                    temperature=config.temperature,
                    max_tokens=512
                )
                reasoning_text = gen_result["reasoning_text"]
                reasoning_id = gen_result["reasoning_id"]
                sentences = gen_result["sentences"]
                
                # Save event
                await TraceEventRepository.create_event(
                    db=db,
                    event_id=uuid.uuid4(),
                    trace_id=trace_id,
                    operator="generate",
                    iteration=iteration,
                    input_summary={"problem": problem.description[:200], "constraints_count": len(constraints_accum)},
                    output_summary={"reasoning_id": reasoning_id, "sentence_count": len(sentences)},
                    model_fingerprint=model_fingerprint(config.model),
                    duration_seconds=gen_result.get("_metadata", {}).get("duration_seconds")
                )
                
                final_reasoning_obj = gen_result
                
            except Exception as e:
                logger.error(f"Generate failed: {str(e)}")
                status = "failed"
                break
            
            # ------------------------------
            # V — Verify
            # ------------------------------
            try:
                ver_result = ver_op.execute(
                    reasoning_text=reasoning_text,
                    reasoning_id=reasoning_id,
                    constraints=constraints_accum
                )
                errors = ver_result["errors"]
                
                # Save event
                await TraceEventRepository.create_event(
                    db=db,
                    event_id=uuid.uuid4(),
                    trace_id=trace_id,
                    operator="verify",
                    iteration=iteration,
                    input_summary={"reasoning_length": len(reasoning_text)},
                    output_summary={"error_count": len(errors)},
                    duration_seconds=ver_result.get("_metadata", {}).get("duration_seconds")
                )
                
            except Exception as e:
                logger.error(f"Verify failed: {str(e)}")
                errors = []
            
            # ------------------------------
            # X — Extract Constraints (if errors found)
            # ------------------------------
            if errors:
                try:
                    ext_result = ext_op.execute(
                        errors=errors,
                        reasoning_text=reasoning_text,
                        verify_threshold=config.verify_threshold,
                        candidate_threshold=config.candidate_threshold
                    )
                    new_constraints = ext_result["constraints"]
                    candidate_constraints = ext_result["candidate_constraints"]
                    
                    # Save constraints to database
                    for constraint in new_constraints:
                        constraint["constraint_id"] = uuid.uuid4()
                        constraint["trace_id"] = trace_id
                        await ConstraintRepository.create_constraint(db=db, **constraint)
                    
                    constraints_accum.extend(new_constraints)
                    
                    # Save event
                    await TraceEventRepository.create_event(
                        db=db,
                        event_id=uuid.uuid4(),
                        trace_id=trace_id,
                        operator="extract",
                        iteration=iteration,
                        input_summary={"error_count": len(errors)},
                        output_summary={"constraints_added": len(new_constraints)},
                        duration_seconds=ext_result.get("_metadata", {}).get("duration_seconds")
                    )
                    
                    # Check constraint cap
                    if len(constraints_accum) >= config.max_constraints:
                        logger.info(f"Max constraints reached: {len(constraints_accum)}")
                        status = "max_constraints"
                        break
                    
                except Exception as e:
                    logger.error(f"Extract failed: {str(e)}")
            
            # ------------------------------
            # O_stab — Stability Check
            # ------------------------------
            try:
                stab_result = stab_op.execute(
                    reasoning_curr=reasoning_text,
                    reasoning_prev=prev_reasoning,
                    threshold=config.similarity_threshold,
                    errors=errors
                )
                is_stable = stab_result["stable"]
                similarity_score = stab_result["score"]
                
                # Save event
                await TraceEventRepository.create_event(
                    db=db,
                    event_id=uuid.uuid4(),
                    trace_id=trace_id,
                    operator="stabilize",
                    iteration=iteration,
                    input_summary={"has_prev": prev_reasoning is not None},
                    output_summary={"stable": is_stable, "score": similarity_score},
                    duration_seconds=stab_result.get("_metadata", {}).get("duration_seconds")
                )
                
                if is_stable:
                    logger.info(f"Converged at iteration {iteration}")
                    status = "converged"
                    break
                    
            except Exception as e:
                logger.error(f"Stabilize failed: {str(e)}")
            
            prev_reasoning = reasoning_text
            
            # Commit after each iteration
            await db.commit()
        
        # Check if max iterations reached
        if iteration == config.max_iterations - 1 and status == "running":
            status = "max_iterations"
            logger.info(f"Max iterations reached: {config.max_iterations}")
        
        # Update trace with final status
        await TraceRepository.update_trace(
            db=db,
            trace_id=trace_id,
            updates={
                "status": status,
                "final_reasoning": reasoning_text if final_reasoning_obj else None,
                "iteration_count": iteration + 1,
                "constraint_count": len(constraints_accum),
                "error_count": len(errors) if 'errors' in locals() else 0
            }
        )
        await db.commit()
        
        # Construct response
        payload = {
            "trace_id": str(trace_id),
            "timestamp": time.time(),
            "proto_version": PROTO_VERSION,
            "status": status,
            "final_reasoning": final_reasoning_obj,
            "constraints": constraints_accum,
            "iteration_count": iteration + 1,
            "model_fingerprint": model_fingerprint(config.model)
        }
        
        payload["node_signature"] = sign_payload(payload)
        
        logger.info(f"ERCP run completed: trace_id={trace_id}, status={status}")
        return payload
        
    except Exception as e:
        logger.error(f"ERCP run failed: {str(e)}", exc_info=True)
        
        # Update trace status to failed
        try:
            await TraceRepository.update_trace(
                db=db,
                trace_id=trace_id,
                updates={"status": "failed"}
            )
            await db.commit()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"ERCP run failed: {str(e)}")


# ============================
# Individual Operator Endpoints
# ============================

@app.post("/ercp/v1/generate")
@limiter.limit("100/minute")
async def api_generate(
    req: GenerateRequest,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(optional_api_key)
):
    """Generate reasoning endpoint (optional authentication, rate limited)."""
    trace_id = UUID(req.trace_id) if req.trace_id else make_trace_id()
    
    try:
        gen_op = get_generate_operator()
        result = gen_op.execute(
            problem_description=req.problem.description,
            constraints=req.constraints,
            **req.gen_config
        )
        
        payload = {
            "trace_id": str(trace_id),
            "model_fingerprint": model_fingerprint(req.gen_config.get("model", DEFAULT_MODEL)),
            "node_signature": sign_payload(result),
            **result
        }
        
        return payload
        
    except Exception as e:
        logger.error(f"Generate failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ercp/v1/verify")
@limiter.limit("100/minute")
async def api_verify(
    req: VerifyRequest,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(optional_api_key)
):
    """Verify reasoning endpoint (optional authentication, rate limited)."""
    trace_id = UUID(req.trace_id) if req.trace_id else make_trace_id()
    
    try:
        ver_op = get_verify_operator()
        result = ver_op.execute(
            reasoning_text=req.reasoning_text,
            reasoning_id=req.reasoning_id,
            constraints=req.constraints,
            **req.verify_config
        )
        
        payload = {
            "trace_id": str(trace_id),
            "errors": result["errors"],
            "error_count": result["error_count"],
            "model_fingerprint": model_fingerprint(req.verify_config.get("model", DEFAULT_MODEL)),
            "node_signature": sign_payload(result)
        }
        
        return payload
        
    except Exception as e:
        logger.error(f"Verify failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ercp/v1/extract_constraints")
@limiter.limit("100/minute")
async def api_extract(
    req: ExtractRequest,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(optional_api_key)
):
    """Extract constraints endpoint (optional authentication, rate limited)."""
    trace_id = UUID(req.trace_id) if req.trace_id else make_trace_id()
    
    try:
        ext_op = get_extract_operator()
        result = ext_op.execute(
            errors=req.errors,
            reasoning_text=req.reasoning_text,
            **req.extract_config
        )
        
        payload = {
            "trace_id": str(trace_id),
            **result,
            "model_fingerprint": model_fingerprint(),
            "node_signature": sign_payload(result)
        }
        
        return payload
        
    except Exception as e:
        logger.error(f"Extract failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ercp/v1/stabilize")
@limiter.limit("100/minute")
async def api_stabilize(
    req: StabilizeRequest,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[str] = Depends(optional_api_key)
):
    """Stabilize (semantic similarity) endpoint (optional authentication, rate limited)."""
    trace_id = UUID(req.trace_id) if req.trace_id else make_trace_id()
    
    try:
        stab_op = get_stabilize_operator()
        result = stab_op.execute(
            reasoning_curr=req.reasoning_curr,
            reasoning_prev=req.reasoning_prev,
            threshold=req.threshold,
            errors=req.errors
        )
        
        payload = {
            "trace_id": str(trace_id),
            **result,
            "node_signature": sign_payload(result)
        }
        
        return payload
        
    except Exception as e:
        logger.error(f"Stabilize failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# Trace Retrieval Endpoint
# ============================

@app.get("/ercp/v1/trace/{trace_id}")
async def get_trace(
    trace_id: str,
    include_events: bool = True,
    include_constraints: bool = True,
    include_errors: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve a trace by ID with all related data."""
    try:
        trace_uuid = UUID(trace_id)
        
        # Get trace with relationships
        trace = await TraceRepository.get_trace(
            db=db,
            trace_id=trace_uuid,
            include_events=include_events,
            include_constraints=include_constraints,
            include_errors=include_errors
        )
        
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
        
        # Convert to response format
        response = {
            "trace_id": str(trace.trace_id),
            "problem_id": trace.problem_id,
            "problem_description": trace.problem_description,
            "status": trace.status,
            "created_at": trace.created_at.isoformat(),
            "updated_at": trace.updated_at.isoformat() if trace.updated_at else None,
            "config": trace.config,
            "final_reasoning": trace.final_reasoning,
            "iteration_count": trace.iteration_count,
            "constraint_count": trace.constraint_count,
            "error_count": trace.error_count
        }
        
        if include_events and trace.events:
            response["events"] = [
                {
                    "event_id": str(event.event_id),
                    "operator": event.operator,
                    "iteration": event.iteration,
                    "timestamp": event.timestamp.isoformat(),
                    "input_summary": event.input_summary,
                    "output_summary": event.output_summary,
                    "duration_seconds": event.duration_seconds,
                    "success": event.success
                }
                for event in trace.events
            ]
        
        if include_constraints and trace.constraints:
            response["constraints"] = [
                {
                    "constraint_id": str(constraint.constraint_id),
                    "type": constraint.type,
                    "priority": constraint.priority,
                    "nl_text": constraint.nl_text,
                    "predicate": constraint.predicate,
                    "confidence": constraint.confidence,
                    "immutable": constraint.immutable
                }
                for constraint in trace.constraints
            ]
        
        if include_errors and trace.errors:
            response["errors"] = [
                {
                    "error_id": str(error.error_id),
                    "type": error.type,
                    "span": error.span,
                    "excerpt": error.excerpt,
                    "confidence": error.confidence,
                    "detected_by": error.detected_by,
                    "evidence": error.evidence
                }
                for error in trace.errors
            ]
        
        return response
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid trace_id format: {trace_id}")
    except Exception as e:
        logger.error(f"Get trace failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ercp/v1/traces")
async def list_traces(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    problem_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List traces with optional filtering."""
    try:
        traces = await TraceRepository.list_traces(
            db=db,
            limit=limit,
            offset=offset,
            status=status,
            problem_id=problem_id
        )
        
        total = await TraceRepository.count_traces(
            db=db,
            status=status,
            problem_id=problem_id
        )
        
        return {
            "traces": [
                {
                    "trace_id": str(trace.trace_id),
                    "problem_id": trace.problem_id,
                    "status": trace.status,
                    "created_at": trace.created_at.isoformat(),
                    "iteration_count": trace.iteration_count,
                    "constraint_count": trace.constraint_count
                }
                for trace in traces
            ],
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List traces failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# Root Endpoint
# ============================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ERCP Protocol Reference Server",
        "version": PROTO_VERSION,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "run": "/ercp/v1/run",
            "generate": "/ercp/v1/generate",
            "verify": "/ercp/v1/verify",
            "extract": "/ercp/v1/extract_constraints",
            "stabilize": "/ercp/v1/stabilize",
            "get_trace": "/ercp/v1/trace/{trace_id}",
            "list_traces": "/ercp/v1/traces"
        }
    }
