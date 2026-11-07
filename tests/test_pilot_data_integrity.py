import json
from pathlib import Path
import numpy as np
import pytest

DATASET_PATH = Path("data/prepared/hopper-medium-v2/dataset.npz")
MANIFEST_PATH = Path("data/prepared/hopper-medium-v2/manifest.json")


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="Pilot data not yet downloaded or prepared")
def test_pilot_hopper_medium_data_integrity():
    with np.load(DATASET_PATH, allow_pickle=False) as data:
        assert "observations" in data
        assert "actions" in data
        assert "returns_to_go" in data
        assert "timesteps" in data
        assert "episode_ids" in data
        assert "is_validation" in data

        n_transitions = len(data["returns_to_go"])
        # Hopper medium dataset has ~1 million transitions (~999k eligible)
        assert n_transitions > 900_000

        # Hopper observation dimension is 11, action dimension is 3
        assert data["observations"].shape == (n_transitions, 11)
        assert data["actions"].shape == (n_transitions, 3)
        assert data["timesteps"].shape == (n_transitions,)

        # Timesteps should be within [0, 1000)
        assert np.all(data["timesteps"] >= 0)
        assert np.all(data["timesteps"] < 1000)

        # Returns-to-go scaling c=1000
        # D4RL Hopper medium maximum episode return is ~3220 -> scaled RTG ~3.22
        max_rtg = float(np.max(data["returns_to_go"]))
        min_rtg = float(np.min(data["returns_to_go"]))
        mean_rtg = float(np.mean(data["returns_to_go"]))

        assert 2.5 <= max_rtg <= 4.0, f"Unexpected max RTG: {max_rtg}"
        assert min_rtg >= -0.5, f"Unexpected min RTG: {min_rtg}"
        assert 0.5 <= mean_rtg <= 2.5, f"Unexpected mean RTG: {mean_rtg}"

        # Train and validation episode separation (no data leakage)
        val_mask = data["is_validation"]
        train_eps = set(data["episode_ids"][~val_mask])
        val_eps = set(data["episode_ids"][val_mask])

        assert len(val_eps) > 0
        assert len(train_eps) > 0
        assert train_eps.isdisjoint(val_eps)

        val_fraction = len(val_eps) / (len(train_eps) + len(val_eps))
        assert 0.08 <= val_fraction <= 0.12, f"Validation fraction expected ~0.10, got {val_fraction}"

    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["audit"]["rtg_eligible_episodes"] > 2000
    assert len(manifest["train_episode_ids"]) + len(manifest["validation_episode_ids"]) > 2000
