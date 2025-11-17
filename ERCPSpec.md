# ERCP Protocol — Spec v1.0  
**Evo-Recursive Constraint Prompting (ERCP)**

A formal, model-agnostic protocol for deterministic, constraint-driven, auditable iterative LLM reasoning.

**Author:** Dr. Heman Mohabeer — EvoLogics AI Lab  
**Document:** Protocol Specification v1.0  
**Status:** Stable Draft

---

# Table of contents
1. Overview  
2. Core concepts & guarantees  
3. API endpoints  
4. JSON schemas  
5. Trace & audit requirements  
6. Determinism & fingerprinting  
7. Verification Oracle contract  
8. Constraint predicate DSL  
9. Typical run lifecycle  
10. Versioning & extensibility  
11. Security & governance notes  
12. Appendix: canonical example

---

# 1. Overview

ERCP standardizes the iterative reasoning loop used to produce stable, high-quality, verifiable outputs from LLMs.

Core operator pipeline:

1. **Generate (G)** – produce reasoning candidate  
2. **Verify (V)** – detect contradictions, errors, ambiguities  
3. **Extract Constraints (X)** – derive constraints from errors  
4. **Stabilize (Oₛ)** – check similarity & absence of errors  
5. **Mutate (M)** – supervisor-level reframe/relax/decompose  

This documents the *REST API*, *schemas*, *stability conditions*, and *audit requirements*.

---

# 2. Core Concepts & Guarantees

**Finite-state termination:**  
Under deterministic decoding and bounded constraint growth, ERCP’s lower-level loop terminates in finite iterations (see theory paper).

**Determinism requirement:**  
Reference implementations *must* support deterministic decoding  
(temperature = 0, top_p = 1.0, greedy or beam with fixed seed).

**Auditability requirements:**  
Every response must include:
- `trace_id`
- `timestamp`
- `proto_version`
- `model_fingerprint`
- `node_signature` (HMAC over canonical JSON)

**Constraint duality:**  
Each constraint must be stored as:
- Human-readable natural language (`nl_text`)
- Machine-checkable predicate (`predicate`)

---

# 3. API Endpoints (base path `/ercp/v1/`)

All endpoints use JSON.  
All responses include audit metadata.

---

## 3.1 `POST /ercp/v1/run`
Run full ERCP loop with given config.

### Request
```json
{
  "trace_id": "optional-uuid",
  "problem": {
    "id": "string",
    "description": "string",
    "metadata": {"domain":"physics","priority":"high"}
  },
  "config": {
    "model": "string",
    "max_iterations": 50,
    "max_constraints": 30,
    "similarity_threshold": 0.95,
    "temperature": 0.0,
    "top_p": 1.0,
    "deterministic": true,
    "verify_threshold": 0.75,
    "candidate_threshold": 0.60
  }
}
```

### Response
```json
{
  "trace_id":"uuid",
  "timestamp":"iso8601",
  "proto_version":"ercp-1.0",
  "status":"converged|infeasible|partial|failed",
  "final_reasoning": {
    "reasoning_id":"uuid",
    "reasoning_text":"string",
    "sentences":[],
    "claims":[]
  },
  "constraints":[{}],
  "trace_events":[{}],
  "utility_score":null,
  "model_fingerprint":"sha256:...",
  "node_signature":"hmac:..."
}
```

---

## 3.2 `POST /ercp/v1/generate`

### Request
```json
{
  "trace_id":"uuid",
  "problem":{},
  "constraints":[],
  "gen_config":{
    "model":"",
    "temperature":0.0,
    "max_tokens":2000,
    "deterministic":true
  }
}
```

### Response
```json
{
  "trace_id":"uuid",
  "reasoning_id":"uuid",
  "reasoning_text":"string",
  "sentences":[ "S1", "S2" ],
  "claims":[{"claim":"...","source":"llm"}],
  "model_fingerprint":"...",
  "node_signature":"..."
}
```

---

## 3.3 `POST /ercp/v1/verify`

### Request
```json
{
  "trace_id":"uuid",
  "reasoning_id":"uuid",
  "reasoning_text":"string",
  "constraints":[],
  "verify_config":{"nli_threshold":0.7,"run_fact_check":true}
}
```

### Response
```json
{
  "trace_id":"uuid",
  "errors":[{}],
  "model_fingerprint":"...",
  "node_signature":"..."
}
```

---

## 3.4 `POST /ercp/v1/extract_constraints`

### Request
```json
{
  "trace_id":"uuid",
  "errors":[{}],
  "reasoning_text":"string",
  "extract_config":{"max_constraints_per_error":2}
}
```

### Response
```json
{
  "trace_id":"uuid",
  "constraints":[{}],
  "candidate_constraints":[{}],
  "model_fingerprint":"...",
  "node_signature":"..."
}
```

---

## 3.5 `POST /ercp/v1/stabilize`

### Request
```json
{
  "trace_id":"uuid",
  "reasoning_prev":"string",
  "reasoning_curr":"string",
  "threshold":0.95
}
```

### Response
```json
{
  "trace_id":"uuid",
  "stable":true,
  "score":0.956,
  "node_signature":"..."
}
```

---

## 3.6 `POST /ercp/v1/mutate`

### Request
```json
{
  "trace_id":"uuid",
  "problem":{},
  "reasoning_text":"string",
  "mutation_strategy":"relax|reframe|decompose",
  "mutation_config":{}
}
```

### Response
```json
{
  "trace_id":"uuid",
  "new_problem":{},
  "new_constraints":[],
  "mutation_notes":"string",
  "node_signature":"..."
}
```

---

## 3.7 `GET /ercp/v1/trace/{trace_id}`

### Response
```json
{
  "trace_id":"uuid",
  "trace_events":[{}],
  "model_fingerprints":[],
  "node_signature":"..."
}
```

---

# 4. JSON Schemas

## 4.1 Constraint Object
```json
{
  "constraint_id":"uuid",
  "type":"predicate|style|factual|temporal|custom",
  "priority":"high|medium|low",
  "nl_text":"string",
  "predicate":{
    "predicate_name":"NoContradiction",
    "args":{"entity":"water.boiling_point"}
  },
  "source":{"detected_by":"nli","error_id":"uuid"},
  "confidence":0.92,
  "immutable":false
}
```

---

## 4.2 Error Object
```json
{
  "error_id":"uuid",
  "type":"contradiction|factual_incorrect|missing_justification|ambiguity|syntax_error",
  "span":[0,12],
  "excerpt":"string",
  "confidence":0.86,
  "detected_by":["nli","rule"],
  "evidence":[{"source":"nli","score":0.86}]
}
```

---

## 4.3 Trace Event
```json
{
  "event_id":"uuid",
  "trace_id":"uuid",
  "timestamp":"iso8601",
  "operator":"generate|verify|extract|stabilize|mutate",
  "input_summary":{},
  "output_summary":{},
  "model_fingerprint":"sha256:...",
  "node_signature":"hmac:..."
}
```

---

# 5. Trace & Audit Requirements

- Append-only logs  
- HMAC-SHA256 signing of each event  
- Include model fingerprint for reproducibility  
- Store decoding parameters (temp, seed, top_p, etc.)

---

# 6. Determinism & Model Fingerprinting

- Deterministic decoding must be supported (`temperature = 0`)  
- Fingerprint = `model_name + version + weights_sha256`  
- If provider is blackbox, include provider metadata + hashed response

---

# 7. Verification Oracle Contract

Must accept:
- `reasoning_text`
- optional `constraints`
- optional `retrieval_context`

Must produce:
- list of error objects with confidence & evidence

Recommended components:
- NLI sentence-pair contradiction detector  
- Rule-based validators  
- Retrieval-backed fact-checker  
- Optional Z3/PySAT checks for predicate validation  

Confidence rule:  
- Auto-add constraints only if `confidence >= verify_threshold`  
- Candidate if `candidate_threshold <= confidence < verify_threshold`

---

# 8. Constraint Predicate DSL

Simple structured predicates:

```json
{
  "predicate_name":"Equal",
  "args":{"left":"water.boiling_point","right":"100","unit":"C"}
}
```

Supported operators initially:
- `Equal`  
- `NotEqual`  
- `LessThan`  
- `GreaterThan`  
- `NoContradiction`  
- `TemporalOrder`  
- `HasJustification`

---

# 9. Typical Run Lifecycle

1. Client calls `/run`.  
2. Server calls `generate`.  
3. Server calls `verify`.  
4. If errors: call `extract_constraints`.  
5. Add constraints meeting threshold.  
6. Re-generate with new constraints.  
7. Stabilize (semantic similarity + no errors).  
8. If stable: return result.  
9. If infeasible: `mutate`, restart.

---

# 10. Versioning & Extensibility

- Include `"proto_version": "ercp-1.0"`  
- Backward-compatible changes allowed in minor versions  
- Breaking changes require major version bump  
- Optional config fields must not break existing clients

---

# 11. Security & Governance

- Token-based auth (JWT/API keys)  
- Optional RBAC for high-priority tasks  
- PII redaction by default  
- Apache-2.0 licensing for code  
- Spec itself can be MIT or Apache

---

# 12. Appendix — Canonical Example

### Input
```json
{
  "problem":{"id":"p1","description":"Why does water boil at different temperatures at different altitudes?"},
  "config":{"model":"evo-41m","max_iterations":10,"similarity_threshold":0.95,"deterministic":true}
}
```

### High-level flow
- Generate: reasoning mentions "boiling point"  
- Verify: missing justification → error  
- Extract: constraint "BoilingPointAtSeaLevel ≈ 100°C"  
- Re-generate: corrected reasoning  
- Stabilize: similarity > threshold  
- Return final trace

---

*End of ERCP Protocol v1.0 Specification*
