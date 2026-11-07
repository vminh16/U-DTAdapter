"""Dataset downloader for D4RL offline benchmarks.

Streams raw HDF5 files directly from official mirrors without requiring
legacy gym or d4rl package installations.
"""

from pathlib import Path
import sys
import urllib.request


D4RL_URLS: dict[str, str] = {
    # Hopper
    "hopper-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
    "hopper-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5",
    # Walker2d
    "walker2d-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
    "walker2d-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5",
    "walker2d-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5",
    # HalfCheetah
    "halfcheetah-medium-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-replay-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah-expert-v2": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5",
}


def download_dataset(
    name: str,
    target_dir: str | Path = "data/raw",
    force: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download an official D4RL dataset in raw HDF5 format.

    Args:
        name: Canonical dataset name (e.g. 'hopper-medium-v2').
        target_dir: Directory where the raw HDF5 file is saved.
        force: If True, re-download even if the file already exists locally.
        chunk_size: Streaming chunk size in bytes (default 1 MB).

    Returns:
        Path to the downloaded HDF5 file.

    Raises:
        ValueError: If dataset name is not registered.
        OSError: If a network or disk error occurs.
    """
    if name not in D4RL_URLS:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {sorted(D4RL_URLS.keys())}")

    url = D4RL_URLS[name]
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    dest_file = target_path / filename

    if dest_file.exists() and not force and dest_file.stat().st_size > 0:
        return dest_file

    tmp_file = dest_file.with_name(f"{filename}.tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "udt-adapter/1.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp_file, "wb") as out_f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)

        # Atomic rename once download is fully complete
        tmp_file.replace(dest_file)
        return dest_file

    except Exception:
        # Clean up temporary file on failure
        if tmp_file.exists():
            tmp_file.unlink()
        raise
