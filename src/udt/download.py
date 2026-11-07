"""Dataset downloader for D4RL offline benchmarks.

Streams raw HDF5 files directly from official mirrors without requiring
legacy gym or d4rl package installations.
"""

from pathlib import Path
import hashlib
import urllib.request
import h5py


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


def download_dataset(
    name: str,
    target_dir: str | Path = "data/raw",
    force: bool = False,
    chunk_size: int = 1024 * 1024,
    timeout: float = 60.0,
    expected_sha256: str | None = None,
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
        # Check if cache is uncorrupted
        if h5py.is_hdf5(dest_file):
            if expected_sha256 is not None:
                h = hashlib.sha256()
                with open(dest_file, "rb") as f:
                    while chunk := f.read(chunk_size):
                        h.update(chunk)
                if h.hexdigest().lower() == expected_sha256.lower():
                    return dest_file
            else:
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

        if expected_sha256 is not None:
            actual_sha256 = hasher.hexdigest().lower()
            if actual_sha256 != expected_sha256.lower():
                raise OSError(
                    f"SHA-256 mismatch for {name}: expected {expected_sha256}, got {actual_sha256}"
                )

        # Atomic rename once verified
        tmp_file.replace(dest_file)
        completed_successfully = True
        return dest_file

    finally:
        # Guarantee cleanup even on KeyboardInterrupt or unhandled exceptions
        if not completed_successfully and tmp_file.exists():
            tmp_file.unlink()
