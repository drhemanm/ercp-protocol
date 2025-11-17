# server/ercp_server.py
"""
ERCP Reference Server — FastAPI Implementation (v1.0 Production-Ready)
Author: Dr. Heman Mohabeer — EvoLogics AI Lab
License: Apache-2.0
"""

import uuid
import time
import hashlib
import hmac
import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from collections import defaultdict
import threading

from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================
# CONFIGURATION
# ============================

# Security: Load from environment with secure defaults
APP_SECRET = os.getenv("ERCP_SECRET_KEY", "").encode() or os.urandom(32)
if len(APP_SECRET) < 32:
    raise ValueError("ERCP_SECRET_KEY must be at least 32 bytes")

API_KEYS = set(os.getenv("ERCP_API_KEYS", "").split(",")) if os.getenv("ERCP_API_KEYS") else set()
PROTO_VERSION = "ercp-1.0"
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB default
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ercp_server")

# ============================
# SECURITY & RATE LIMITING
# ============================

security = HTTPBearer(auto_error=False)

# Simple in-memory rate limiter (replace with Redis in production)
rate_limit_store: Dict[str, List[float]] = defaultdict(list)
rate_limit_lock = threading.Lock()

def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit."""
    now = time.time()
    with rate_limit_lock:
        # Clean old requests
        rate_limit_store[client_id] = [
            req_time for req_time in rate_limit_store[client_id]
            if now - req_time < RATE_LIMIT_WINDOW
        ]

        if len(rate_limit_store[client_id]) >= RATE_LIMIT_REQUESTS:
            return False

        rate_limit_store[client_id].append(now)
        return True

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API token and apply rate limiting."""
    if API_KEYS and credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials"
        )

    if API_KEYS and credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    client_id = credentials.credentials if credentials else "anonymous"

    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s"
        )

    return client_id

# ============================
# TRACE STORAGE
# ============================

# In-memory trace storage (replace with database in production)
trace_storage: Dict[str, Dict] = {}
trace_storage_lock = threading.Lock()

def store_trace(trace_id: str, trace_data: Dict) -> None:
    """Store trace data securely."""
    with trace_storage_lock:
        trace_storage[trace_id] = trace_data

def get_trace(trace_id: str) -> Optional[Dict]:
    """Retrieve trace data."""
    with trace_storage_lock:
        return trace_storage.get(trace_id)


# ------------------------------------
# Utilities
# ------------------------------------

def make_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def get_iso_timestamp() -> str:
    """Return current timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sign_payload(payload: dict) -> str:
    """
    Compute HMAC-SHA256 signature of a payload.
    Ensures auditability and tamper detection.
    """
    # Remove signature if present to avoid circular dependency
    payload_copy = {k: v for k, v in payload.items() if k != "node_signature"}
    raw = json.dumps(payload_copy, sort_keys=True, default=str).encode()
    signature = hmac.new(APP_SECRET, raw, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{signature}"


def verify_signature(payload: dict, signature: str) -> bool:
    """Verify HMAC signature of a payload."""
    expected = sign_payload(payload)
    return hmac.compare_digest(expected, signature)


def model_fingerprint(model_name: str = "local-llm", model_version: str = "v1.0") -> str:
    """
    Generate model fingerprint for reproducibility.
    In production, include actual model weights hash.
    """
    fingerprint_data = f"{model_name}:{model_version}:{PROTO_VERSION}"
    hash_digest = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    return f"sha256:{hash_digest}"


def sanitize_text(text: str) -> str:
    """
    Basic PII redaction placeholder.
    In production, use NLP-based PII detection.
    """
    # TODO: Implement proper PII redaction (emails, SSN, credit cards, etc.)
    return text


def validate_constraint_predicate(predicate: Dict) -> bool:
    """Validate constraint predicate structure."""
    if not isinstance(predicate, dict):
        return False

    predicate_name = predicate.get("predicate_name")
    valid_predicates = {
        "Equal", "NotEqual", "LessThan", "GreaterThan",
        "NoContradiction", "TemporalOrder", "HasJustification"
    }

    return predicate_name in valid_predicates


# ============================
# Pydantic Models (Schemas with Validation)
# ============================

class Problem(BaseModel):
    id: Optional[str] = None
    description: str
    metadata: Optional[dict] = {}

    @validator('description')
    def validate_description(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Problem description cannot be empty")
        if len(v) > 10000:
            raise ValueError("Problem description too long (max 10000 chars)")
        return sanitize_text(v)

    @validator('id')
    def validate_id(cls, v):
        if v and len(v) > 100:
            raise ValueError("Problem ID too long (max 100 chars)")
        return v


class Config(BaseModel):
    model: str = "local-llm"
    max_iterations: int = 20
    max_constraints: int = 30
    similarity_threshold: float = 0.95
    temperature: float = 0.0
    deterministic: bool = True
    verify_threshold: float = 0.75
    candidate_threshold: float = 0.60

    @validator('max_iterations')
    def validate_max_iterations(cls, v):
        if v < 1 or v > 1000:
            raise ValueError("max_iterations must be between 1 and 1000")
        return v

    @validator('max_constraints')
    def validate_max_constraints(cls, v):
        if v < 0 or v > 500:
            raise ValueError("max_constraints must be between 0 and 500")
        return v

    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @validator('verify_threshold', 'candidate_threshold')
    def validate_thresholds(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("thresholds must be between 0.0 and 1.0")
        return v


class RunRequest(BaseModel):
    trace_id: Optional[str] = None
    problem: Problem
    config: Config


class GenerateRequest(BaseModel):
    trace_id: Optional[str]
    problem: Problem
    constraints: list
    gen_config: dict


class VerifyRequest(BaseModel):
    trace_id: Optional[str]
    reasoning_id: str
    reasoning_text: str
    constraints: list
    verify_config: dict


class ExtractRequest(BaseModel):
    trace_id: Optional[str]
    errors: list
    reasoning_text: str
    extract_config: dict


class StabilizeRequest(BaseModel):
    trace_id: Optional[str]
    reasoning_prev: Optional[str]
    reasoning_curr: str
    threshold: float


class MutateRequest(BaseModel):
    trace_id: Optional[str]
    problem: Problem
    reasoning_text: str
    mutation_strategy: str
    mutation_config: dict


# ============================
# FastAPI App
# ============================

app = FastAPI(
    title="ERCP Protocol Reference Server",
    version="1.0",
    description="Production-ready implementation for ERCP protocol (spec v1.0)"
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted host middleware in production
if os.getenv("ENVIRONMENT", "development") == "production":
    allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": PROTO_VERSION,
        "timestamp": get_iso_timestamp()
    }


# ==============================================================
# CORE OPERATOR IMPLEMENTATIONS
# ==============================================================

def generate_reasoning(problem_text: str, constraints: list, config: dict) -> dict:
    """
    Generate reasoning with constraint-aware prompting.

    In production: Replace with actual LLM call (OpenAI, Anthropic, local model, etc.)
    Ensure deterministic decoding (temperature=0) for reproducibility.
    """
    logger.info(f"Generating reasoning with {len(constraints)} constraints")

    # Build constraint-aware prompt
    constraint_text = ""
    if constraints:
        constraint_text = "\n\nConstraints to satisfy:\n" + "\n".join([
            f"- {c.get('nl_text', c.get('constraint_id'))}"
            for c in constraints[:10]  # Limit to avoid prompt bloat
        ])

    # Simulated reasoning (replace with actual LLM)
    sample_text = (
        f"Reasoning for: {problem_text}\n\n"
        "Water boils at lower temperatures at higher altitudes due to decreased atmospheric pressure. "
        "At sea level, the standard atmospheric pressure is approximately 101.325 kPa, "
        "and water boils at around 100°C (212°F). "
        "As altitude increases, atmospheric pressure decreases, which lowers the boiling point. "
        "For example, at 2000 meters above sea level, water boils at approximately 93°C."
        f"{constraint_text}"
    )

    sentences = [
        "Water boils at lower temperatures at higher altitudes due to decreased atmospheric pressure.",
        "At sea level, the standard atmospheric pressure is approximately 101.325 kPa.",
        "Water boils at around 100°C (212°F) at sea level.",
        "As altitude increases, atmospheric pressure decreases, which lowers the boiling point.",
        "At 2000 meters above sea level, water boils at approximately 93°C."
    ]

    claims = [
        {"claim": "Boiling point decreases with altitude", "source": "llm", "confidence": 0.95},
        {"claim": "Sea-level boiling is ~100°C", "source": "llm", "confidence": 0.98},
        {"claim": "Standard atmospheric pressure is 101.325 kPa", "source": "llm", "confidence": 0.97}
    ]

    return {
        "reasoning_id": str(uuid.uuid4()),
        "reasoning_text": sample_text,
        "sentences": sentences,
        "claims": claims,
    }


def verify_reasoning(reasoning_text: str, constraints: list, config: dict) -> list:
    """
    Verify reasoning for errors and constraint violations.

    In production: Use NLI models, fact-checkers, rule-based validators.
    Recommended: MNLI, RoBERTa-large-MNLI, or similar for contradiction detection.
    """
    logger.info(f"Verifying reasoning against {len(constraints)} constraints")

    errors = []
    verify_threshold = config.get("verify_threshold", 0.75)

    # Rule-based checks (example)
    sentences = reasoning_text.split(". ")

    # Check for basic quality issues
    if len(reasoning_text.strip()) < 50:
        errors.append({
            "error_id": str(uuid.uuid4()),
            "type": "missing_justification",
            "span": [0, len(reasoning_text)],
            "excerpt": reasoning_text[:100],
            "confidence": 0.9,
            "detected_by": ["rule"],
            "evidence": [{"source": "rule", "score": 0.9}]
        })

    # Check for constraint violations
    for constraint in constraints:
        predicate = constraint.get("predicate", {})
        nl_text = constraint.get("nl_text", "")

        # Simple keyword-based check (replace with NLI in production)
        if predicate.get("predicate_name") == "NoContradiction":
            # In production: use NLI model to detect contradictions
            pass

        # Validate numeric constraints
        if predicate.get("predicate_name") in ["Equal", "LessThan", "GreaterThan"]:
            # In production: extract values and validate
            pass

    # Simulate occasional error detection for testing
    # Remove this in production when real verification is implemented

    logger.info(f"Found {len(errors)} errors")
    return errors


def extract_constraints_from_errors(errors: list, reasoning_text: str, config: dict) -> dict:
    """
    Extract structured constraints from detected errors.

    In production: Use meta-prompting or fine-tuned extraction model.
    """
    logger.info(f"Extracting constraints from {len(errors)} errors")

    constraints = []
    candidate_constraints = []
    max_per_error = config.get("max_constraints_per_error", 2)

    for error in errors:
        error_type = error.get("type")
        confidence = error.get("confidence", 0.0)

        # Generate constraints based on error type
        if error_type == "contradiction":
            constraint = {
                "constraint_id": str(uuid.uuid4()),
                "type": "predicate",
                "priority": "high",
                "nl_text": "Avoid contradictions in reasoning",
                "predicate": {
                    "predicate_name": "NoContradiction",
                    "args": {}
                },
                "source": {"detected_by": "nli", "error_id": error.get("error_id")},
                "confidence": confidence,
                "immutable": False
            }

            if confidence >= config.get("verify_threshold", 0.75):
                constraints.append(constraint)
            else:
                candidate_constraints.append(constraint)

        elif error_type == "missing_justification":
            constraint = {
                "constraint_id": str(uuid.uuid4()),
                "type": "predicate",
                "priority": "medium",
                "nl_text": "Provide justification for all claims",
                "predicate": {
                    "predicate_name": "HasJustification",
                    "args": {"claim": error.get("excerpt", "")}
                },
                "source": {"detected_by": "rule", "error_id": error.get("error_id")},
                "confidence": confidence,
                "immutable": False
            }

            if confidence >= config.get("verify_threshold", 0.75):
                constraints.append(constraint)
            else:
                candidate_constraints.append(constraint)

    logger.info(f"Extracted {len(constraints)} constraints, {len(candidate_constraints)} candidates")

    return {
        "constraints": constraints[:max_per_error * len(errors)],
        "candidate_constraints": candidate_constraints
    }


def semantic_stability(prev: Optional[str], curr: str, threshold: float) -> dict:
    """
    Check semantic stability using text similarity.

    In production: Use sentence transformers, BERTScore, or similar.
    Example: sentence-transformers/all-MiniLM-L6-v2
    """
    logger.info("Computing semantic stability")

    if prev is None:
        return {"stable": False, "score": 0.0}

    # Simple character-level similarity (replace with embeddings in production)
    # In production, use: sentence_transformers.SentenceTransformer + cosine similarity
    prev_words = set(prev.lower().split())
    curr_words = set(curr.lower().split())

    if not prev_words or not curr_words:
        return {"stable": False, "score": 0.0}

    intersection = prev_words.intersection(curr_words)
    union = prev_words.union(curr_words)
    jaccard_similarity = len(intersection) / len(union) if union else 0.0

    # This is a very basic approximation - use proper embeddings in production
    score = jaccard_similarity
    stable = score >= threshold

    logger.info(f"Stability score: {score:.3f}, threshold: {threshold}, stable: {stable}")

    return {"stable": stable, "score": score}


def mutate_problem(problem: Problem, reasoning_text: str, strategy: str, config: dict) -> dict:
    """
    Supervisor-level mutation for problem reformulation.

    Strategies:
    - relax: Remove or weaken some constraints
    - reframe: Reformulate the problem
    - decompose: Break into sub-problems
    """
    logger.info(f"Mutating problem with strategy: {strategy}")

    new_problem = problem.copy()
    new_constraints = []
    mutation_notes = ""

    if strategy == "relax":
        # Remove lowest-priority constraints
        mutation_notes = "Relaxed constraint requirements to enable convergence"
        new_constraints = []  # Will be filtered by caller

    elif strategy == "reframe":
        # Reformulate problem description
        new_problem.description = f"Simplified: {problem.description}"
        mutation_notes = "Reframed problem for clarity"

    elif strategy == "decompose":
        # Break into sub-problems
        new_problem.metadata["decomposed"] = True
        mutation_notes = "Decomposed into simpler sub-problem"
    else:
        mutation_notes = f"Unknown strategy: {strategy}, no mutation applied"

    logger.info(f"Mutation complete: {mutation_notes}")

    return {
        "new_problem": new_problem,
        "new_constraints": new_constraints,
        "mutation_notes": mutation_notes
    }


# ==============================================================
# API ENDPOINTS IMPLEMENTATION
# ==============================================================

@app.post("/ercp/v1/run")
async def run_ercp(
    req: RunRequest,
    client_id: str = Depends(verify_token)
):
    """
    Execute full ERCP reasoning loop with convergence detection.
    """
    trace_id = req.trace_id or make_trace_id()
    problem = req.problem
    config = req.config

    logger.info(f"Starting ERCP run {trace_id} for client {client_id}")

    try:
        prev_reasoning = None
        constraints_accum = []
        trace_events = []
        final_status = "converged"
        iterations_run = 0

        for iteration in range(config.max_iterations):
            iterations_run = iteration + 1
            logger.info(f"Iteration {iteration + 1}/{config.max_iterations}")

            # ------------------------------
            # G — Generate
            # ------------------------------
            try:
                gen_out = generate_reasoning(problem.description, constraints_accum, config.dict())
                reasoning_text = gen_out["reasoning_text"]

                trace_events.append({
                    "event_id": str(uuid.uuid4()),
                    "trace_id": trace_id,
                    "timestamp": get_iso_timestamp(),
                    "operator": "generate",
                    "iteration": iteration,
                    "input_summary": {"constraints_count": len(constraints_accum)},
                    "output_summary": {"reasoning_length": len(reasoning_text)},
                    "model_fingerprint": model_fingerprint(config.model)
                })
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                final_status = "failed"
                break

            # ------------------------------
            # V — Verify
            # ------------------------------
            try:
                errors = verify_reasoning(reasoning_text, constraints_accum, config.dict())

                trace_events.append({
                    "event_id": str(uuid.uuid4()),
                    "trace_id": trace_id,
                    "timestamp": get_iso_timestamp(),
                    "operator": "verify",
                    "iteration": iteration,
                    "input_summary": {"reasoning_length": len(reasoning_text)},
                    "output_summary": {"errors_found": len(errors)},
                    "model_fingerprint": model_fingerprint(config.model)
                })
            except Exception as e:
                logger.error(f"Verification failed: {e}")
                final_status = "failed"
                break

            # ------------------------------
            # X — Extract Constraints
            # ------------------------------
            if errors:
                try:
                    ex_out = extract_constraints_from_errors(errors, reasoning_text, config.dict())
                    new_constraints = ex_out["constraints"]

                    constraints_accum.extend(new_constraints)

                    trace_events.append({
                        "event_id": str(uuid.uuid4()),
                        "trace_id": trace_id,
                        "timestamp": get_iso_timestamp(),
                        "operator": "extract",
                        "iteration": iteration,
                        "input_summary": {"errors_count": len(errors)},
                        "output_summary": {"constraints_added": len(new_constraints)},
                        "model_fingerprint": model_fingerprint(config.model)
                    })

                    if len(constraints_accum) >= config.max_constraints:
                        logger.warning(f"Constraint cap reached: {len(constraints_accum)}")
                        final_status = "partial"
                        break
                except Exception as e:
                    logger.error(f"Constraint extraction failed: {e}")
                    final_status = "failed"
                    break

            # ------------------------------
            # O_stab — Stability
            # ------------------------------
            try:
                stab = semantic_stability(prev_reasoning, reasoning_text, config.similarity_threshold)

                trace_events.append({
                    "event_id": str(uuid.uuid4()),
                    "trace_id": trace_id,
                    "timestamp": get_iso_timestamp(),
                    "operator": "stabilize",
                    "iteration": iteration,
                    "input_summary": {"has_previous": prev_reasoning is not None},
                    "output_summary": {"stable": stab["stable"], "score": stab["score"]},
                    "model_fingerprint": model_fingerprint(config.model)
                })

                if stab["stable"] and not errors:
                    logger.info(f"Converged at iteration {iteration + 1}")
                    final_status = "converged"
                    break

                prev_reasoning = reasoning_text
            except Exception as e:
                logger.error(f"Stability check failed: {e}")
                final_status = "failed"
                break

        # Construct final response
        payload = {
            "trace_id": trace_id,
            "timestamp": get_iso_timestamp(),
            "proto_version": PROTO_VERSION,
            "status": final_status,
            "iterations_run": iterations_run,
            "final_reasoning": gen_out if 'gen_out' in locals() else None,
            "constraints": constraints_accum,
            "trace_events": trace_events,
            "utility_score": None,  # Can be computed if needed
            "model_fingerprint": model_fingerprint(config.model)
        }

        payload["node_signature"] = sign_payload(payload)

        # Store trace for retrieval
        store_trace(trace_id, payload)

        logger.info(f"ERCP run {trace_id} completed with status: {final_status}")
        return payload

    except Exception as e:
        logger.error(f"ERCP run {trace_id} failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ERCP run failed: {str(e)}"
        )


@app.post("/ercp/v1/generate")
async def api_generate(
    req: GenerateRequest,
    client_id: str = Depends(verify_token)
):
    """Generate reasoning with constraint-aware prompting."""
    logger.info(f"Generate request from client {client_id}")

    try:
        out = generate_reasoning(req.problem.description, req.constraints, req.gen_config)
        payload = {
            "trace_id": req.trace_id or make_trace_id(),
            "timestamp": get_iso_timestamp(),
            "model_fingerprint": model_fingerprint(req.gen_config.get("model", "local-llm")),
            **out
        }
        payload["node_signature"] = sign_payload(payload)
        return payload
    except Exception as e:
        logger.error(f"Generate failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/ercp/v1/verify")
async def api_verify(
    req: VerifyRequest,
    client_id: str = Depends(verify_token)
):
    """Verify reasoning for errors and constraint violations."""
    logger.info(f"Verify request from client {client_id}")

    try:
        errs = verify_reasoning(req.reasoning_text, req.constraints, req.verify_config)
        payload = {
            "trace_id": req.trace_id or make_trace_id(),
            "timestamp": get_iso_timestamp(),
            "errors": errs,
            "model_fingerprint": model_fingerprint(req.verify_config.get("model", "local-llm"))
        }
        payload["node_signature"] = sign_payload(payload)
        return payload
    except Exception as e:
        logger.error(f"Verify failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@app.post("/ercp/v1/extract_constraints")
async def api_extract(
    req: ExtractRequest,
    client_id: str = Depends(verify_token)
):
    """Extract structured constraints from detected errors."""
    logger.info(f"Extract constraints request from client {client_id}")

    try:
        out = extract_constraints_from_errors(req.errors, req.reasoning_text, req.extract_config)
        payload = {
            "trace_id": req.trace_id or make_trace_id(),
            "timestamp": get_iso_timestamp(),
            **out,
            "model_fingerprint": model_fingerprint()
        }
        payload["node_signature"] = sign_payload(payload)
        return payload
    except Exception as e:
        logger.error(f"Extract failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Constraint extraction failed: {str(e)}"
        )


@app.post("/ercp/v1/stabilize")
async def api_stabilize(
    req: StabilizeRequest,
    client_id: str = Depends(verify_token)
):
    """Check semantic stability between reasoning iterations."""
    logger.info(f"Stabilize request from client {client_id}")

    try:
        stab = semantic_stability(req.reasoning_prev, req.reasoning_curr, req.threshold)
        payload = {
            "trace_id": req.trace_id or make_trace_id(),
            "timestamp": get_iso_timestamp(),
            **stab
        }
        payload["node_signature"] = sign_payload(payload)
        return payload
    except Exception as e:
        logger.error(f"Stabilize failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stability check failed: {str(e)}"
        )


@app.post("/ercp/v1/mutate")
async def api_mutate(
    req: MutateRequest,
    client_id: str = Depends(verify_token)
):
    """
    Supervisor-level mutation for problem reformulation.
    Strategies: relax, reframe, decompose
    """
    logger.info(f"Mutate request from client {client_id} with strategy: {req.mutation_strategy}")

    if req.mutation_strategy not in ["relax", "reframe", "decompose"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mutation strategy: {req.mutation_strategy}"
        )

    try:
        out = mutate_problem(req.problem, req.reasoning_text, req.mutation_strategy, req.mutation_config)
        payload = {
            "trace_id": req.trace_id or make_trace_id(),
            "timestamp": get_iso_timestamp(),
            **out
        }
        payload["node_signature"] = sign_payload(payload)
        return payload
    except Exception as e:
        logger.error(f"Mutate failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mutation failed: {str(e)}"
        )


@app.get("/ercp/v1/trace/{trace_id}")
async def get_trace_endpoint(
    trace_id: str,
    client_id: str = Depends(verify_token)
):
    """
    Retrieve full audit trace for a given trace_id.
    Provides complete transparency and auditability.
    """
    logger.info(f"Trace retrieval request for {trace_id} from client {client_id}")

    trace_data = get_trace(trace_id)

    if trace_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}"
        )

    # Return trace with additional metadata
    response = {
        "trace_id": trace_id,
        "retrieved_at": get_iso_timestamp(),
        "trace_data": trace_data
    }

    return response
