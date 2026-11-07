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
├── requirements.txt             # Core runtime dependencies
├── requirements-dev.txt         # Development and testing dependencies
└── pyproject.toml               # Package build metadata
```

## Quickstart

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

### 2. Run Tests

```bash
pytest tests/test_data.py tests/test_intervention.py -v
```

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
