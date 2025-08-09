# Coursework: Classical Machine Learning

## Requirements for Reproducibility Infrastructure

### Environment Lock
- Save a list of packages from rocm to the repo (pip freeze > requirements.lock.txt)
- Mark "read-only" in the README (for reference only, not for installation)

### Pre-commit Hooks
- Add pre-commit configuration with:
  - ruff (linting)
  - black (code formatting)
  - isort (import sorting)
  - nbstripout (Jupyter notebook output removal)
- Configure .pre-commit-config.yaml
- Enable git hook

### Experiment Tracking
- Add experiments/registry.jsonl
- Add experiment logging utility (append_jsonl)

### Continuous Integration
- Set up simple CI (GitHub Actions or local script ci.sh)
- Include linters + smoke tests on small sample

### Artifacts
- requirements.lock.txt
- .pre-commit-config.yaml
- .github/workflows/ci.yml (or ci.sh)
- src/utils/io.py (append_jsonl)

### Readiness Criteria
- `make info/eda/train_*` pass
- Pre-commit is working
- At least 1 smoke test has been added