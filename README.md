# udt-adapter: Test-Time Intervention for Frozen Decision Transformers

Research workspace for **U-DTVerifier**: an inference-time RTG calibration and disagreement-penalized action reranking framework on frozen Decision Transformers (DT).

## Overview

Offline Reinforcement Learning via Sequence Modeling (Decision Transformer) conditions action generation on desired return-to-go (RTG). When evaluated on out-of-distribution (OOD) return prompts or perturbed states, autoregressive generation can drift into ungrounded trajectories. 

`udt-adapter` implements test-time intervention modules:
1. **RTG Calibration / Capping**: Constrains return budget tracking within empirically supported state-time envelopes.
2. **Disagreement-Penalized Reranking**: Evaluates candidate actions using ensemble heads, penalizing high epistemic uncertainty.

## Directory Layout

```
.
├── src/
│   └── udt/                     # Core research library (validation, audit, intervention)
├── baselines/
│   └── decision_transformer/    # Original Decision Transformer codebase (Chen et al., 2021)
├── docs/
│   ├── research/                # Mathematical critique, RRD, and experiment protocols
│   └── agents/                  # Autonomous agent workflow guidelines and tracking configs
├── tests/                       # Unit and integration test suites
├── scripts/                     # Thin experiment entrypoints
├── requirements.txt             # All runtime and test dependencies
└── pyproject.toml               # Package build metadata only
```

## Quickstart

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Verified development environment: Python 3.10.12, torch 2.5.1+cu121,
numpy 2.2.6, gymnasium 1.2.3, mujoco 3.3.7, transformers 4.18.0.

### 2. Run Tests

```bash
pytest -q
```

### 3. Experiment Scripts

Every script is thin: it validates arguments, calls one `udt` function, and
prints a summary. Run them from the repository root after the editable install.

```bash
# Data
python scripts/download_data.py hopper-medium-v2 --target-dir data/raw
python scripts/audit_data.py data/raw/hopper-medium-v2.hdf5 --horizon 1000 \
  --starts-at-reset --timeout-semantics horizon --output data/audit.json
python scripts/prepare_data.py data/raw/hopper-medium-v2.hdf5 --horizon 1000 \
  --starts-at-reset --timeout-semantics horizon \
  --output data/prepared/hopper-medium-v2 --dataset-id hopper-medium-v2

# Models
python scripts/train_dt.py --data data/prepared/hopper-medium-v2/dataset.npz \
  --manifest data/prepared/hopper-medium-v2/manifest.json
python scripts/train_r.py --data data/prepared/hopper-medium-v2/dataset.npz \
  --horizon 1000 --output checkpoints/module_r --mode state_time
python scripts/train_u.py --data data/prepared/hopper-medium-v2/dataset.npz \
  --horizon 1000 --output checkpoints/module_u --num-models 5

# Evaluation (one shared rollout path for every variant)
python scripts/evaluate.py --variant B0 --checkpoint checkpoints/decision_transformer/model.pt \
  --env Hopper-v4 --episodes 10 --horizon 1000 --target-return 1800 \
  --parity-output results/b0_parity.json --results-output results/b0_evaluation.json
python scripts/evaluate.py --variant B3 --capper checkpoints/module_r/return_capper.pt \
  --parity-output results/b3_parity.json --results-output results/b3_evaluation.json

# Readiness is variant-scoped: B0/B3 do not require Module U artifacts.
python scripts/check_readiness.py --manifest data/prepared/hopper-medium-v2/manifest.json \
  --dataset data/prepared/hopper-medium-v2/dataset.npz \
  --return-checkpoint checkpoints/module_r/return_capper.pt \
  --parity results/b3_parity.json --variants B0,B3 --json
```

`check_readiness.py` exits non-zero while any blocker remains (missing
provenance, split leakage, missing real-DT parity, incomplete
bootstrap/normalization metadata for the requested variants, unavailable R/U
components, or untraceable checkpoints). A real pilot is only reportable when
that command returns `"ready": true` against a real frozen DT checkpoint.
Latency is measured with `udt.profiling.measure_latency` /
`measure_controller_latency`, which report mean/p50/p95 plus hardware and
software metadata.

## Research Documentation

- [Research Requirements Document (RRD)](docs/research/U-DTVerifier_RRD.md)
- [Mathematical Review & Bounds](docs/research/U-DTVerifier_Mathematical_Review.md)
- [Experiment Protocol](docs/research/EXPERIMENT.md)

## Citation & Baseline Attribution

Original Decision Transformer baseline code is preserved under `baselines/decision_transformer/` and cited from:
```bibtex
@article{chen2021decisiontransformer,
  title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author={Lili Chen and Kevin Lu and Aravind Rajeswaran and Kimin Lee and Aditya Grover and Michael Laskin and Pieter Abbeel and Aravind Srinivas and Igor Mordatch},
  journal={arXiv preprint arXiv:2106.01345},
  year={2021}
}
```
