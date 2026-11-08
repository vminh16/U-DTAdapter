"""Dataset downloader for D4RL offline benchmarks.

Streams raw HDF5 files directly from official mirrors without requiring
legacy gym or d4rl package installations.
"""

from pathlib import Path
import hashlib
import json
import urllib.request
import h5py

from udt.provenance import compute_file_sha256


D4RL_URLS: dict[str, str] = {
    # Hopper
    "hopper-medium-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
    "hopper-expert-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5",
    # Walker2d
    "walker2d-medium-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
    "walker2d-medium-replay-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5",
    "walker2d-expert-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5",
    # HalfCheetah
    "halfcheetah-medium-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-replay-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah-expert-v2": "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5",
}


REQUIRED_DATASET_FIELDS = ("observations", "actions", "rewards", "terminals")


def compute_sha256(path: str | Path) -> str:
    """Backward-compatible alias for the shared provenance hash."""
    return compute_file_sha256(path)


def validate_dataset_file(path: str | Path) -> dict:
    """Validate required fields and basic shapes in an HDF5 dataset.

    A syntactically valid HDF5 container is not automatically a D4RL dataset.
    Optional fields (``next_observations``, ``timeouts``) are checked for
    consistency when present but are not required here; their provenance is
    handled by :mod:`udt.data`.
    """
    target = Path(path)
    try:
        with h5py.File(target, "r") as handle:
            missing = [name for name in REQUIRED_DATASET_FIELDS if name not in handle]
            if missing:
                raise ValueError(f"Dataset missing required HDF5 fields: {missing}")
            for name in REQUIRED_DATASET_FIELDS:
                if not isinstance(handle[name], h5py.Dataset):
                    raise ValueError(f"{name} must be an HDF5 dataset, not a group")

            obs = handle["observations"]
            actions = handle["actions"]
            rewards = handle["rewards"]
            terminals = handle["terminals"]
            if obs.ndim != 2 or obs.shape[0] < 1 or obs.shape[1] < 1:
                raise ValueError(f"observations must have shape (N, state_dim), got {obs.shape}")
            n = obs.shape[0]
            if actions.ndim != 2 or actions.shape[0] != n or actions.shape[1] < 1:
                raise ValueError(
                    f"actions must have shape ({n}, act_dim), got {actions.shape}"
                )
            if rewards.shape != (n,):
                raise ValueError(f"rewards must have shape ({n},), got {rewards.shape}")
            if terminals.shape != (n,):
                raise ValueError(f"terminals must have shape ({n},), got {terminals.shape}")
            observed_shape = obs.shape
            for optional in ("next_observations", "timeouts"):
                if optional in handle:
                    dataset = handle[optional]
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError(f"{optional} must be an HDF5 dataset, not a group")
                    if optional == "next_observations" and dataset.shape != observed_shape:
                        raise ValueError(
                            f"next_observations must have shape {observed_shape}, "
                            f"got {dataset.shape}"
                        )
                    if optional == "timeouts" and dataset.shape != (n,):
                        raise ValueError(f"timeouts must have shape ({n},), got {dataset.shape}")
            return {
                "transitions": int(n),
                "state_dim": int(obs.shape[1]),
                "action_dim": int(actions.shape[1]),
                "fields": sorted(handle.keys()),
            }
    except OSError as exc:
        raise ValueError(f"Invalid HDF5 dataset {target}: {exc}") from exc


def _write_provenance(path, name, url, dest_file, sha256, dataset_info, *, reason):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": name,
        "url": url,
        "path": str(dest_file),
        "source_sha256": sha256,
        "size_bytes": dest_file.stat().st_size,
        "dataset_info": dataset_info,
        "provenance_reason": reason,
        "checksum_source": "computed_sha256",
    }
    target.write_text(json.dumps(payload, indent=2))


def download_dataset(
    name: str,
    target_dir: str | Path = "data/raw",
    force: bool = False,
    chunk_size: int = 1024 * 1024,
    timeout: float = 60.0,
    expected_sha256: str | None = None,
    provenance_path: str | Path | None = None,
) -> Path:
    """Download an official D4RL dataset in raw HDF5 format.

    Args:
        name: Canonical dataset name (e.g. 'hopper-medium-v2').
        target_dir: Directory where the raw HDF5 file is saved.
        force: If True, re-download even if the file already exists locally.
        chunk_size: Streaming chunk size in bytes (default 1 MB).
        timeout: Socket timeout in seconds for HTTP requests.
        expected_sha256: Optional expected SHA-256 hex digest to verify integrity.

    Returns:
        Path to the downloaded HDF5 file.

    Raises:
        ValueError: If dataset name is not registered.
        OSError: If a network error occurs, download is truncated, format is invalid,
                 or SHA-256 does not match.
    """
    if name not in D4RL_URLS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {sorted(D4RL_URLS.keys())}")

    url = D4RL_URLS[name]
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    dest_file = target_path / filename

    if dest_file.exists() and not force and dest_file.stat().st_size > 0:
        # A cached file is reusable only when it is HDF5 *and* has a valid
        # dataset schema. A format-valid but empty/foreign HDF5 is not a dataset.
        cache_valid = h5py.is_hdf5(dest_file)
        cache_info: dict | None = None
        if cache_valid:
            try:
                cache_info = validate_dataset_file(dest_file)
            except ValueError:
                cache_valid = False
        if cache_valid:
            cache_hash = compute_sha256(dest_file)
            if expected_sha256 is None or cache_hash.lower() == expected_sha256.lower():
                if provenance_path is not None:
                    _write_provenance(
                        provenance_path, name, url, dest_file, cache_hash, cache_info,
                        reason="cache_hit",
                    )
                return dest_file

    tmp_file = dest_file.with_name(f"{filename}.tmp")
    completed_successfully = False

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "udt-adapter/1.0"})
        hasher = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp_file, "wb") as out_f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)

        if downloaded == 0:
            raise OSError(f"Downloaded 0 bytes from {url}")

        if total_size > 0 and downloaded < total_size:
            raise OSError(
                f"Incomplete download for {name}: received {downloaded}/{total_size} bytes"
            )

        if not h5py.is_hdf5(tmp_file):
            raise OSError(f"Downloaded file {tmp_file} is not a valid HDF5 file")

        actual_sha256 = hasher.hexdigest().lower()
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise OSError(
                f"SHA-256 mismatch for {name}: expected {expected_sha256}, got {actual_sha256}"
            )

        try:
            dataset_info = validate_dataset_file(tmp_file)
        except ValueError as exc:
            raise OSError(f"Downloaded HDF5 for {name} is not a valid dataset: {exc}") from exc

        # Atomic rename once verified
        tmp_file.replace(dest_file)
        completed_successfully = True
        if provenance_path is not None:
            _write_provenance(
                provenance_path, name, url, dest_file, actual_sha256, dataset_info,
                reason="downloaded",
            )
        return dest_file

    finally:
        # Guarantee cleanup even on KeyboardInterrupt or unhandled exceptions
        if not completed_successfully and tmp_file.exists():
            tmp_file.unlink()
