# Repository Reset & Environment Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the forked Decision Transformer repository into a clean, standalone research workspace for `udt-adapter` (U-DTVerifier) with a fresh Git history, `requirements.txt` environment management, and verified unit test baseline.

**Architecture:** Purge upstream git history while preserving the original DT code under `baselines/decision_transformer/`. Establish a standardized virtual environment using explicit `requirements.txt` and `requirements-dev.txt` manifests, install editable package bindings, verify data and intervention test suites, and seal with a clean initial commit on branch `main`.

**Tech Stack:** Python 3.10+, pip, venv, Git, pytest, numpy, h5py.

**Spec:** User specification to reset git author history, rename/rebrand project identity to `udt-adapter` (U-DTVerifier), and configure environment using `requirements.txt` for reproducibility.

## Global Constraints

- Python version: >= 3.10.
- No system-site-packages pollution in `.venv`.
- Original DT paper code preserved in `baselines/decision_transformer/`.
- Research documentation in `docs/research/` preserved intact.
- Core dependencies: `numpy>=1.23,<3`, `h5py>=3.8,<4`, `pytest>=7,<9`.

## Review Focus

1. Upstream git commits leaking into new repository (must verify `git log` contains only the initial commit).
2. Missing or unignored build artifacts like `.venv/` or `.egg-info/` in git staging.
3. Module resolution failure when importing `udt` outside package root.
4. Breaking unit test assertions in `tests/test_data.py` or `tests/test_intervention.py`.
5. Stray CLI test failures before `cli.py` implementation (CLI tests to be scoped out until `cli.py` is written).

---

### Task 1: Reset Git History and Establish Repository Identity

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `.git/` (reinitialize)

**Interfaces:**
- Produces: Clean Git repository initialized on branch `main` with project README reflecting `udt-adapter` / U-DTVerifier.

- [ ] **Step 1: Update README.md to reflect new repository identity**

Replace root `README.md` with documentation for `udt-adapter` (U-DTVerifier), explaining:
- Project title: `udt-adapter: Test-Time Intervention for Frozen Decision Transformers`
- Overview of U-DTVerifier research foundation
- Directory layout: `src/udt` (adapter code), `baselines/decision_transformer` (original paper code), `docs/research/` (RRD, math review, experiment protocol)
- Quickstart instructions using `requirements.txt`

- [ ] **Step 2: Ensure .gitignore covers all build and environment artifacts**

Verify and ensure `.gitignore` contains:
```gitignore
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
build/
dist/
data/
runs/
checkpoints/
wandb/
.env
```

- [ ] **Step 3: Remove upstream Git history and reinitialize fresh Git repository**

Run:
```bash
rm -rf .git
git init -b main
```
Verify: `git branch --show-current` outputs `main` and `git status` shows clean untracked state without remotes.

---

### Task 2: Environment Management with requirements.txt and Clean .venv

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `.venv/` (recreated)

**Interfaces:**
- Consumes: `pyproject.toml` dependency declarations.
- Produces: Standalone `requirements.txt` and `requirements-dev.txt`, plus an active, clean Python virtual environment.

- [ ] **Step 1: Create `requirements.txt`**

Write `requirements.txt` with base runtime dependencies:
```text
numpy>=1.23,<3
h5py>=3.8,<4
```

- [ ] **Step 2: Create `requirements-dev.txt`**

Write `requirements-dev.txt` referencing `requirements.txt`:
```text
-r requirements.txt
pytest>=7,<9
```

- [ ] **Step 3: Recreate clean virtual environment**

Run:
```bash
rm -rf .venv
python3 -m venv .venv
```
Verify: `.venv/bin/python --version` confirms Python 3.10+ and `grep "include-system-site-packages = false" .venv/pyvenv.cfg` confirms site isolation.

- [ ] **Step 4: Install dependencies and editable package into .venv**

Run:
```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e . --no-deps
```
Verify: `.venv/bin/pip list` shows `numpy`, `h5py`, `pytest`, and editable `udt-adapter` installed.

---

### Task 3: Verification of Unit Tests and Initial Baseline Commit

**Files:**
- Test: `tests/test_data.py`
- Test: `tests/test_intervention.py`

**Interfaces:**
- Consumes: Installed `udt` package, `pytest`.
- Produces: Passing unit tests for data pipeline and intervention modules, plus initial Git commit on `main`.

- [ ] **Step 1: Run unit tests for data pipeline and intervention primitives**

Run:
```bash
.venv/bin/pytest tests/test_data.py tests/test_intervention.py -v
```
Expected: PASS (all tests in `test_data.py` and `test_intervention.py` succeed).

- [ ] **Step 2: Verify package importability**

Run:
```bash
.venv/bin/python -c "import udt; import udt.data; import udt.intervention; print(f'udt v{udt.__version__} successfully imported')"
```
Expected: PASS with output `udt v0.1.0 successfully imported`.

- [ ] **Step 3: Stage repository files and make initial commit**

Run:
```bash
git add .
git commit -m "feat: initial clean repository setup for udt-adapter"
```
Verify: `git log --oneline` shows exactly 1 commit: `feat: initial clean repository setup for udt-adapter`.
