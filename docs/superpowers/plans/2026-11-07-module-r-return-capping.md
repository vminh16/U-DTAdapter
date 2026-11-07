# Module R (Return Capping & Calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Module R (test-time Return-to-Go calibration & capping) for frozen Decision Transformers, including expectile loss, state-time return MLP, global/time-only baselines (B2, B3, B9), and the training pipeline with full test coverage and an independent subagent review.

**Architecture:** A standalone PyTorch-based module `src/udt/return_cap.py` providing asymmetric expectile regression (`ExpectileLoss`), a lightweight state-time MLP (`ReturnCapMLP`) and baseline variants (`GlobalReturnCap`, `TimeOnlyCap`, `StateOnlyCap`), connected via a clean inference interface `ReturnCapper` that interfaces directly with `ReturnBudget` in `src/udt/intervention.py`. Trained on suffix return labels produced by `src/udt/data.py`.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, NumPy, pytest.

**Spec:** `docs/research/U-DTVerifier_RRD.md` Section 3.1 & 3.2; `docs/research/U-DTVerifier_Mathematical_Review.md` Section 1, 2, 4; `docs/research/EXPERIMENT.md` Baselines B2, B3, B9.

## Global Constraints

- Python version: >= 3.10.
- Parameter budget for $g_\psi$: < 300k parameters (< 1.2 MB weights).
- Return scaling: $c = 1000$ (undiscounted $\gamma = 1.0$).
- State-time conditioning: input must be $(z_t, t/H)$ where $z_t = (o_t - \mu)/\sigma$ and $t/H \in [0, 1]$.
- Strict decoupling: raw budget recurrence $B_{t+1}^{raw} = B_t^{raw} - r_t/c$ must never be modified by the clipped token $B_t$.
- English docstrings and comments on critical mathematical logic.

## Review Focus

1. Asymmetric loss gradient inversion: ensure $\tau=0.9$ penalizes under-estimation ($y > \hat{y}$) with weight $\tau$ and over-estimation with weight $1-\tau$.
2. Boundary horizon stability: at $t \ge H$, relative time $t/H \ge 1.0$, return prediction must not produce NaNs or exploding values.
3. Raw budget recurrence isolation: verify that calling `ReturnCapper.step()` returns the clipped token but leaves `budget.raw` untouched for the next reward decrement.
4. Support for all baseline configurations: B2 (global scalar), B3 (state-time), B9 (time-only), and state-only ablation.
5. Suffix return validation: reject datasets containing censored/unsupported suffix returns.

---

### Task 1: Environment & PyTorch Dependency Setup

**Files:**
- Modify: `requirements.txt`
- Modify: `.venv/`

**Interfaces:**
- Produces: Installed CUDA-enabled PyTorch inside `.venv` and updated dependency specification.

- [ ] **Step 1: Update requirements.txt to include PyTorch**

Add `torch>=2.0` to `requirements.txt`.

- [ ] **Step 2: Install PyTorch into .venv**

Run:
```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Verify: `.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add torch dependency to requirements.txt"
```

---

### Task 2: Core Mathematical Primitives: ExpectileLoss & GlobalReturnCap

**Files:**
- Create: `src/udt/return_cap.py`
- Test: `tests/test_return_cap.py`

**Interfaces:**
- Produces:
  - `ExpectileLoss(tau: float = 0.9, reduction: str = "mean")`
  - `compute_global_expectile(returns: np.ndarray, tau: float = 0.9) -> float`
  - `GlobalReturnCap(cap_value: float)`

- [ ] **Step 1: Write failing tests for ExpectileLoss and GlobalReturnCap**

In `tests/test_return_cap.py`:
- `test_expectile_loss_symmetric_tau_half`: at $\tau=0.5$, $\text{loss} = 0.5 \times \text{MSE}$.
- `test_expectile_loss_asymmetry`: for error $e = y - \hat{y}$, loss when $e > 0$ is $\tau / (1 - \tau)$ times loss when $e < 0$.
- `test_expectile_loss_rejects_invalid_tau`: raises `ValueError` for $\tau \le 0$ or $\tau \ge 1$.
- `test_compute_global_expectile`: finds the scalar $c$ satisfying $\mathbb{E}[|\tau - \mathbb{I}(Y < c)|(Y - c)] = 0$.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_return_cap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'udt.return_cap'`.

- [ ] **Step 3: Implement ExpectileLoss and GlobalReturnCap in `src/udt/return_cap.py`**

Implement:
```python
class ExpectileLoss(nn.Module):
    def __init__(self, tau: float = 0.9, reduction: str = "mean"): ...
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # diff = target - input
        # weight = torch.abs(self.tau - (diff < 0).float())
        # loss = weight * (diff ** 2)
```
and `compute_global_expectile` using root finding / iterative expectile updates on numpy arrays, plus `GlobalReturnCap`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_return_cap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/udt/return_cap.py tests/test_return_cap.py
git commit -m "feat(return_cap): add ExpectileLoss and GlobalReturnCap"
```

---

### Task 3: Neural Return Models (B3, B9, Ablation) & ReturnCapper Interface

**Files:**
- Modify: `src/udt/return_cap.py`
- Test: `tests/test_return_cap.py`

**Interfaces:**
- Consumes: `ExpectileLoss`, `ReturnBudget` (from `src/udt/intervention.py`).
- Produces:
  - `ReturnCapMLP(state_dim: int, hidden_dim: int = 128, num_layers: int = 3, mode: str = "state_time")`
    `mode` $\in$ `["state_time", "time_only", "state_only"]`.
  - `ReturnCapper(model: nn.Module | GlobalReturnCap, horizon: int, margin: float = 0.0, scale: float = 1000.0)`
    Method: `get_cap(state: np.ndarray, step: int) -> float`
    Method: `step(budget: ReturnBudget, state: np.ndarray, step: int) -> float`

- [ ] **Step 1: Write failing tests for ReturnCapMLP and ReturnCapper**

In `tests/test_return_cap.py`:
- `test_return_cap_mlp_forward_shapes`: test output shape `(batch, 1)` for 1D and 2D batch inputs across modes (`state_time`, `time_only`, `state_only`).
- `test_return_cap_mlp_parameter_count`: assert total parameters < 300,000.
- `test_return_capper_clips_budget_without_mutating_raw`: create `ReturnBudget(raw=1.5)`, call `capper.step(budget, state, step=10)` with cap=1.0, verify returned token is 1.0 while `budget.raw` remains 1.5.
- `test_return_capper_respects_safety_margin`: verify token is $\min(B_t^{raw}, \text{cap} + \delta)$.
- `test_return_capper_handles_horizon_boundary`: step $\ge H$ sets relative time to 1.0 without crash.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_return_cap.py -v`
Expected: FAIL with `cannot import name 'ReturnCapMLP'`.

- [ ] **Step 3: Implement ReturnCapMLP and ReturnCapper**

In `src/udt/return_cap.py`:
- Implement `ReturnCapMLP` with configurable `mode`:
  - `state_time`: input dimension `state_dim + 1` (normalized state $z_t$ concatenated with $t/H$).
  - `time_only`: input dimension `1` ($t/H$).
  - `state_only`: input dimension `state_dim` ($z_t$).
- Implement `ReturnCapper` with `get_cap()` and `step(budget, state, step)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_return_cap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/udt/return_cap.py tests/test_return_cap.py
git commit -m "feat(return_cap): add ReturnCapMLP and ReturnCapper interface"
```

---

### Task 4: Training Pipeline for Module R (`train_return_cap`)

**Files:**
- Modify: `src/udt/return_cap.py`
- Test: `tests/test_return_cap.py`

**Interfaces:**
- Consumes: Dataset dictionary from `src/udt/data.py:prepare_dataset`.
- Produces:
  - `train_return_cap(dataset: dict, state_dim: int, horizon: int, tau: float = 0.9, mode: str = "state_time", lr: float = 1e-3, epochs: int = 5, batch_size: int = 256, device: str = "cpu") -> ReturnCapper`
  - Training metrics dict containing train/validation loss history and empirical quantile coverage.

- [ ] **Step 1: Write failing test for train_return_cap**

In `tests/test_return_cap.py`:
- `test_train_return_cap_synthetic_dataset`: generate synthetic prepared dataset (using `src/udt/data.py` fixtures), call `train_return_cap`, verify:
  - Validation loss decreases over training.
  - Returned object is an initialized `ReturnCapper`.
  - Empirical coverage $\mathbb{P}(Y \le \text{cap}) \ge \tau - 0.15$ on validation set.
- `test_train_return_cap_global_baseline`: verify mode `"global"` returns a `ReturnCapper` wrapping `GlobalReturnCap` without neural training.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_return_cap.py -k "test_train_return_cap" -v`
Expected: FAIL with `cannot import name 'train_return_cap'`.

- [ ] **Step 3: Implement train_return_cap in `src/udt/return_cap.py`**

Implement `train_return_cap` with:
- PyTorch DataLoader over `observations`, `returns_to_go`, `timesteps`.
- Adam optimizer and `ExpectileLoss`.
- Validation loop tracking `val_loss` and empirical coverage.
- Early stopping / best model checkpoint restoration.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_return_cap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/udt/return_cap.py tests/test_return_cap.py
git commit -m "feat(return_cap): add train_return_cap training pipeline"
```

---

### Task 5: End-to-End Evaluation Integration & Independent Subagent Review

**Files:**
- Create: `src/udt/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `ReturnCapper`, `ReturnBudget`.
- Produces:
  - `evaluate_with_budget(env_step_fn, capper, initial_target, horizon)` simulating the closed-loop evaluation step and tracking `clipping_frequency`, `raw_returns`, `tokens_used`.

- [ ] **Step 1: Write failing test for evaluate_with_budget**

In `tests/test_evaluation.py`:
- Test mock closed-loop rollout where target is inflated:
  - Verify that when `initial_target` is inflated, `clipping_frequency > 0`.
  - Verify `tokens_used` never exceed `cap + margin`.
  - Verify when capper is None, `clipping_frequency == 0.0`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'udt.evaluation'`.

- [ ] **Step 3: Implement evaluate_with_budget in `src/udt/evaluation.py`**

Implement closed-loop stepping loop that hooks `ReturnCapper` and logs telemetry (`clipping_frequency`, target trajectory, actual rewards).

- [ ] **Step 4: Run full test suite to verify everything passes**

Run: `.venv/bin/pytest -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/udt/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): add evaluate_with_budget rollout harness"
```

- [ ] **Step 6: Dispatch independent code reviewer subagent**

Dispatch a general-purpose subagent to review the entire git diff against `U-DTVerifier_RRD.md` and `U-DTVerifier_Mathematical_Review.md`.
Reviewer checks:
- Expectile loss gradient correctness.
- State-time formulation conformance.
- Raw budget recurrence isolation.
- Code quality and English docstrings on critical math.
Fix any issues flagged by the reviewer.
