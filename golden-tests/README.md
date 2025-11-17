# Golden Tests

This folder will contain curated problems and expected outputs for reproducibility and CI.

Structure (proposed):
- golden-tests/
  - piqa_sample_01.json
  - contradiction_case_01.json
  - expected_outputs/

Add test cases as JSON with fields:
- problem.id
- problem.description
- expected_final_reasoning (optional)
- expected_constraints (optional)
