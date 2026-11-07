# Pilot Dataset Download & Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a dependency-light D4RL dataset downloader and HDF5/NPZ loader for the offline dataset pipeline, and download/audit/prepare the pilot dataset `hopper-medium-v2`.

**Architecture:** A standalone download module `src/udt/download.py` fetching official HDF5 datasets directly from the Berkeley mirror with streaming verification and caching; coupled with a unified `load_dataset` entrypoint in `src/udt/data.py` that parses `.hdf5` and `.npz` into validated transition arrays while strictly rejecting pickle files.

**Tech Stack:** Python 3.10+, NumPy, h5py, urllib.request, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-dataset-download-pilot-design.md`, `docs/research/EXPERIMENT.md` Section 2.

## Global Constraints

- Python version: >= 3.10.
- Raw storage path: `data/raw/` (ignored by git).
- Prepared storage path: `data/prepared/` (ignored by git).
- Target dataset for pilot: `hopper-medium-v2` (~152 MB).
- Security constraint: Strict rejection of `.pkl` files with descriptive error message.
- Zero extra heavy dependencies: no `gym<0.26`, no `d4rl`, no `mujoco_py`.

## Review Focus

1. Partial or interrupted downloads: ensuring interrupted HTTP downloads are cleaned up or not left as corrupt `.hdf5` files.
2. Missing or non-standard HDF5 keys: ensuring `timeouts` defaults to zeros if omitted in older schemas, and observations/actions are verified.
3. Pickle injection prevention: ensuring files named `.pkl` or containing pickle headers are rejected before any unpickling occurs.
4. Idempotent downloads: ensuring re-running the download does not re-download 150MB over the network unless `force=True`.
5. Exact horizon compliance: ensuring D4RL Gym MuJoCo horizon $H=1000$ and timeout semantics match `audit_dataset` expectations.

---

### Task 1: Safe File Loader (`load_dataset`) in `src/udt/data.py`

**Files:**
- Modify: `src/udt/data.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Produces: `load_dataset(path: str | Path) -> dict[str, np.ndarray]`
  - Reads `.hdf5`, `.h5`, and `.npz`.
  - Rejects `.pkl` with `ValueError("Pickle format (.pkl) is not supported; use NPZ or HDF5.")`.
  - Returns dictionary with validated arrays: `observations`, `next_observations`, `actions`, `rewards`, `terminals`, `timeouts`.

- [ ] **Step 1: Write failing tests for load_dataset**

In `tests/test_data.py`:
- `test_load_dataset_npz`: create a temporary `.npz` file with standard keys, verify `load_dataset` loads expected arrays.
- `test_load_dataset_hdf5`: create a temporary `.hdf5` file using `h5py`, verify `load_dataset` extracts all datasets.
- `test_load_dataset_rejects_pickle`: attempt to load a file with `.pkl` extension or pickle data, verify `ValueError` mentioning "NPZ or HDF5".
- `test_load_dataset_missing_file`: verify `FileNotFoundError` when file does not exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_data.py -k "test_load_dataset" -v`
Expected: FAIL with `cannot import name 'load_dataset'`.

- [ ] **Step 3: Implement `load_dataset` in `src/udt/data.py`**

Implement:
```python
def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    if p.suffix.lower() == ".pkl":
        raise ValueError("Pickle format (.pkl) is not supported; use NPZ or HDF5.")
    if p.suffix.lower() in (".hdf5", ".h5"):
        # Read keys using h5py, convert to numpy arrays, handle timeouts
        ...
    elif p.suffix.lower() == ".npz":
        # Read keys using np.load(..., allow_pickle=False)
        ...
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}. Expected .hdf5, .h5, or .npz")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_data.py -k "test_load_dataset" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/udt/data.py tests/test_data.py
git commit -m "feat(data): add load_dataset supporting HDF5 and NPZ formats"
```

---

### Task 2: Dataset Downloader Module (`src/udt/download.py`)

**Files:**
- Create: `src/udt/download.py`
- Create: `tests/test_download.py`

**Interfaces:**
- Produces:
  - `D4RL_URLS: dict[str, str]` (registry mapping dataset IDs to official Berkeley URLs).
  - `download_dataset(name: str, target_dir: str | Path = "data/raw", force: bool = False, chunk_size: int = 1024 * 1024) -> Path`

- [ ] **Step 1: Write failing tests for download_dataset**

In `tests/test_download.py`:
- `test_download_dataset_unknown_name`: raises `ValueError` for unknown dataset name.
- `test_download_dataset_mocked_http`: mock `urllib.request.urlopen`, verify content is downloaded to target directory.
- `test_download_dataset_idempotency`: verify that if destination file exists and is valid, `download_dataset` does not re-download unless `force=True`.
- `test_download_dataset_cleans_up_on_failure`: simulate network error mid-download, verify partial file is deleted and not left corrupt.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'udt.download'`.

- [ ] **Step 3: Implement `src/udt/download.py`**

Implement URL dictionary and `download_dataset` with atomic write (`.tmp` file renamed upon completion) and progress logging.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_download.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/udt/download.py tests/test_download.py
git commit -m "feat(download): add download_dataset with atomic streaming and caching"
```

---

### Task 3: Live Pilot Download, Audit, and Preparation (`hopper-medium-v2`)

**Files:**
- Modify: `data/raw/` (untracked download)
- Modify: `data/prepared/` (untracked prepared dataset)
- Create: `tests/test_pilot_data_integrity.py`

**Interfaces:**
- Consumes: `download_dataset`, `load_dataset`, `audit_dataset`, `prepare_dataset`.
- Produces: Verified `data/raw/hopper_medium-v2.hdf5` and `data/prepared/hopper-medium-v2/dataset.npz` with `manifest.json`.

- [ ] **Step 1: Download live pilot dataset**

Run python script to download `hopper-medium-v2`:
```bash
.venv/bin/python -c "from udt.download import download_dataset; p = download_dataset('hopper-medium-v2', target_dir='data/raw'); print(f'Downloaded to {p}, size: {p.stat().st_size} bytes')"
```
Expected: `data/raw/hopper_medium-v2.hdf5` (~152 MB).

- [ ] **Step 2: Execute audit and prepare on live dataset**

Run python script to audit and prepare:
```bash
.venv/bin/python -c "from pathlib import Path; from udt.data import load_dataset, audit_dataset, prepare_dataset; data = load_dataset('data/raw/hopper_medium-v2.hdf5'); rep = audit_dataset(data, horizon=1000, timeout_semantics='horizon'); print('Audit:', rep['rtg_eligible_episodes'], 'eligible episodes,', rep['rtg_eligible_transitions'], 'transitions'); arrays, manifest = prepare_dataset(data, horizon=1000, timeout_semantics='horizon'); out_dir = Path('data/prepared/hopper-medium-v2'); out_dir.mkdir(parents=True, exist_ok=True); import numpy as np, json; np.savez(out_dir / 'dataset.npz', **arrays); (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2)); print('Saved prepared dataset successfully.')"
```
Verify: `data/prepared/hopper-medium-v2/dataset.npz` and `manifest.json` are generated.

- [ ] **Step 3: Write test verifying pilot dataset integrity**

In `tests/test_pilot_data_integrity.py`:
- Decorate with `@pytest.mark.skipif(not Path("data/prepared/hopper-medium-v2/dataset.npz").exists(), reason="Pilot data not yet downloaded")`.
- Assert `returns_to_go` has length > 900,000.
- Assert `is_validation` splits episodes without overlap.
- Assert max return is consistent with D4RL Hopper-medium scores (~3200 unscaled, ~3.2 scaled).

- [ ] **Step 4: Run test suite**

Run: `.venv/bin/pytest tests/test_pilot_data_integrity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pilot_data_integrity.py
git commit -m "test: add pilot dataset integrity verification test"
```
