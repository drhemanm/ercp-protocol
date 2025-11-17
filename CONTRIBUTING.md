# Contributing to ERCP-Protocol

Thanks for contributing. Follow these rules to keep the repo professional.

## Code style
- Python: Black format (recommended).
- Use type hints and Pydantic schemas for public interfaces.

## Tests
- Add unit tests under `tests/`
- Golden tests go into `golden-tests/` (see README there).

## Pull Requests
- Branch from `main` into `feature/<short-desc>`.
- Open PR with description, test results, and demo GIF if applicable.
- CI will run tests and linting.

## Security
- Do NOT commit secrets. Use `.env` or secret store.
- If you find a security issue, contact the project lead privately:
  heman@evologics.ai
