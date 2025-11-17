# ERCP Protocol

**Evo-Recursive Constraint Prompting (ERCP) — Protocol v1.0**

A model-agnostic, auditable runtime & protocol for provable, self-correcting LLM reasoning.
ERCP formalizes iterative constraint extraction, verification, and bi-level supervision to produce
stable, high-quality reasoning traces with convergence guarantees.

This repository contains:
- The ERCP Protocol specification (`ERCPSpec.md`)
- Reference implementation (FastAPI) in `/server`
- SDKs (Python + JS) in `/sdk`
- Golden tests and examples in `/golden-tests`
- Documentation and the ERCP theory whitepaper in `/docs`

---

## Quick links
- Spec: `ERCPSpec.md` (protocol endpoints, JSON schemas, operators)
- Theory: `docs/ERCP_v2_Theory_Paper_FINAL.pdf`
- Reference server: `server/ercp_server.py`
- Python SDK: `sdk/python/ercp_client.py`
- Golden tests: `golden-tests/`

---

## Why ERCP?

ERCP introduces a rigorous, auditable approach to LLM reasoning:
- **Operator algebra**: `generate`, `verify`, `extract_constraints`, `stabilize`, `mutate`.
- **Convergence guarantees**: provable termination under reasonable assumptions.
- **Constraint-first**: explicit constraints expressed as both NL and machine predicates.
- **Auditability**: signed traces, model fingerprints, and reproducible runs.

ERCP is intended for researchers, engineers, and production systems that require
robust, interpretable, and auditable reasoning behavior from LLMs.

---

## Quickstart (local prototype)

1. Clone repo:
```bash
git clone git@github.com:<your-org>/ercp-protocol.git
cd ercp-protocol
