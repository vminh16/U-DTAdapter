"""One safe checkpoint-loading policy shared by every U-DT artifact loader."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


def load_checkpoint_payload(
    path: str | Path,
    *,
    weights_only: bool = True,
    allow_unsafe_legacy: bool = False,
) -> dict[str, Any]:
    """Load a JSON or PyTorch checkpoint, rejecting unsafe pickle by default."""
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if checkpoint.suffix.lower() == ".json":
        payload = json.loads(checkpoint.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{checkpoint} does not contain a JSON object")
        return payload

    if not weights_only and not allow_unsafe_legacy:
        raise ValueError(
            "Loading checkpoints with weights_only=False is unsafe and prohibited by default. "
            "Pass allow_unsafe_legacy=True explicitly to opt into loading trusted legacy checkpoints."
        )
    if allow_unsafe_legacy:
        warnings.warn(
            "Loading checkpoint with unsafe deserialization (weights_only=False). "
            "Only do this for trusted legacy files.",
            UserWarning,
            stacklevel=2,
        )

    import torch

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=allow_unsafe_legacy is False,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint} does not contain a dictionary payload")
    return payload
