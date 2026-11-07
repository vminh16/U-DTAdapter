# Design: Pilot Dataset Download & Ingestion (`hopper-medium-v2`)

**Date:** 2026-11-07  
**Status:** Proposed  
**Scope:** Small Change (Pilot Data Acquisition & Pipeline Verification)

---

## 1. Intent & Context

To evaluate and verify Module R (Return Capping) and Decision Transformer baselines on real offline reinforcement learning benchmarks without depending on fragile legacy packages (`d4rl`, `gym<0.26`, `mujoco_py`), we need a lightweight, reproducible data acquisition and ingestion pipeline.

The pilot dataset is **`hopper-medium-v2`** (~152 MB) as specified in `docs/research/EXPERIMENT.md` Section 2.

---

## 2. Approaches Considered

1. **Direct HTTP Stream Download from Berkeley HDF5 Mirror (Chosen):**
   - Download the official raw `.hdf5` file directly from `http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5`.
   - Read directly using `h5py` (already in `.venv` and `requirements.txt`).
   - *Pros:* Zero external C++ dependencies, 100% reproducible, fast (< 1 min download), compatible with existing `src/udt/data.py`.
   - *Cons:* Requires maintaining URL map for D4RL datasets (trivial dictionary).

2. **Legacy `d4rl.get_dataset()` (Rejected):**
   - Requires installing `gym`, `d4rl`, `mujoco-py`, and system MuJoCo binaries.
   - *Pros:* Standard D4RL API call.
   - *Cons:* Extremely fragile build process, breaks modern Python 3.10+ wheels, outputs unsafe `.pkl` files which are explicitly forbidden by `tests/test_cli.py`.

---

## 3. Architecture & Interfaces

### 3.1 Dataset Downloader (`src/udt/download.py`)
- **Dataset Registry:** Mapping canonical environment names (`hopper-medium-v2`, `walker2d-medium-v2`, etc.) to remote URLs and expected keys.
- **Function:** `download_dataset(name: str, target_dir: Path | str = "data/raw", force: bool = False) -> Path`
  - Checks if file already exists in `data/raw/`.
  - Streams HTTP download with progress reporting (chunk size 1 MB).
  - Validates that the downloaded file is a readable HDF5 file containing required fields.

### 3.2 Data Loading Support (`src/udt/data.py:load_dataset`)
- **Function:** `load_dataset(path: Path | str) -> dict[str, np.ndarray]`
  - Supports `.hdf5` / `.h5` and `.npz`.
  - Explicitly rejects `.pkl` (pickle) with actionable security/compatibility error message.
  - Extracts standard keys: `observations`, `next_observations` (or computes from observations), `actions`, `rewards`, `terminals`, `timeouts`.

### 3.3 Storage Layout
- Raw data: `data/raw/hopper_medium-v2.hdf5` (anchored in root `.gitignore`).
- Prepared data: `data/prepared/hopper-medium-v2/` containing `dataset.npz` and `manifest.json`.

---

## 4. Verification Plan

1. Unit tests in `tests/test_download.py`:
   - Mock HTTP download with integrity check.
   - Reject invalid/unknown dataset identifiers.
   - Idempotency (skip re-download if file exists unless `force=True`).
2. Integration test in `tests/test_data.py`:
   - `load_dataset` correctly reads HDF5 and NPZ formats.
   - Rejects pickle files with informative error.
3. Live Pilot Execution:
   - Download `hopper-medium-v2.hdf5`.
   - Audit with `udt.data.audit_dataset(horizon=1000)`.
   - Prepare with `udt.data.prepare_dataset(horizon=1000)` and verify split and RTG statistics.
