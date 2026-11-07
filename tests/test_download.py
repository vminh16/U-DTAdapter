import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import h5py
import numpy as np
import pytest

from udt.download import D4RL_URLS, download_dataset


def make_dummy_hdf5_bytes():
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset("observations", data=np.zeros((10, 4)))
        f.create_dataset("actions", data=np.zeros((10, 2)))
        f.create_dataset("rewards", data=np.zeros(10))
        f.create_dataset("terminals", data=np.zeros(10))
    return bio.getvalue()


def test_download_dataset_unknown_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        download_dataset("non-existent-v2")


def test_download_dataset_mocked_http(tmp_path):
    fake_content = make_dummy_hdf5_bytes()

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [fake_content, b""]
    mock_resp.headers.get.return_value = str(len(fake_content))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out_path = download_dataset("hopper-medium-v2", target_dir=tmp_path)
        assert out_path.exists()
        assert out_path.read_bytes() == fake_content
        assert mock_urlopen.call_count == 1


def test_download_dataset_idempotency(tmp_path):
    dest = tmp_path / "hopper_medium-v2.hdf5"
    dest.write_bytes(make_dummy_hdf5_bytes())

    # Should not invoke urlopen because file exists and force=False
    with patch("urllib.request.urlopen") as mock_urlopen:
        out_path = download_dataset("hopper-medium-v2", target_dir=tmp_path, force=False)
        assert out_path == dest
        mock_urlopen.assert_not_called()

    # With force=True, it should re-download
    new_content = make_dummy_hdf5_bytes()
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [new_content, b""]
    mock_resp.headers.get.return_value = str(len(new_content))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out_path = download_dataset("hopper-medium-v2", target_dir=tmp_path, force=True)
        assert out_path.read_bytes() == new_content
        mock_urlopen.assert_called_once()


def test_download_dataset_cleans_up_on_failure(tmp_path):
    mock_resp = MagicMock()
    # Emulate partial stream followed by an exception
    mock_resp.read.side_effect = [b"partial data", OSError("Network disconnected")]
    mock_resp.headers.get.return_value = "1000"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match="Network disconnected"):
            download_dataset("hopper-medium-v2", target_dir=tmp_path)

    # Neither final file nor leftover .tmp file should exist
    assert not (tmp_path / "hopper_medium-v2.hdf5").exists()
    assert not (tmp_path / "hopper_medium-v2.hdf5.tmp").exists()


def test_download_dataset_cleans_up_on_keyboard_interrupt(tmp_path):
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"chunk", KeyboardInterrupt("User cancelled")]
    mock_resp.headers.get.return_value = "1000"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(KeyboardInterrupt):
            download_dataset("hopper-medium-v2", target_dir=tmp_path)

    assert not (tmp_path / "hopper_medium-v2.hdf5").exists()
    assert not (tmp_path / "hopper_medium-v2.hdf5.tmp").exists()


def test_download_dataset_detects_incomplete_download(tmp_path):
    mock_resp = MagicMock()
    # Premature EOF: header says 1000 bytes, stream gives 10 bytes
    mock_resp.read.side_effect = [b"1234567890", b""]
    mock_resp.headers.get.return_value = "1000"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match="Incomplete download"):
            download_dataset("hopper-medium-v2", target_dir=tmp_path)

    assert not (tmp_path / "hopper_medium-v2.hdf5").exists()


def test_download_dataset_validates_hdf5_format(tmp_path):
    # Non-HDF5 payload (e.g. HTML error page with 200 OK)
    html_payload = b"<html><body>Server Error</body></html>"
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [html_payload, b""]
    mock_resp.headers.get.return_value = str(len(html_payload))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match="not a valid HDF5"):
            download_dataset("hopper-medium-v2", target_dir=tmp_path)

    assert not (tmp_path / "hopper_medium-v2.hdf5").exists()


def test_download_dataset_zero_bytes_rejected(tmp_path):
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b""]
    mock_resp.headers.get.return_value = "0"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match="0 bytes"):
            download_dataset("hopper-medium-v2", target_dir=tmp_path)


def test_download_dataset_sha256_verification(tmp_path):
    import hashlib
    content = make_dummy_hdf5_bytes()
    expected_hash = hashlib.sha256(content).hexdigest()

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [content, b""]
    mock_resp.headers.get.return_value = str(len(content))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    # Mismatch hash should raise OSError
    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match="SHA-256 mismatch"):
            download_dataset("hopper-medium-v2", target_dir=tmp_path, expected_sha256="deadbeef" * 8)

    # Correct hash should pass
    mock_resp.read.side_effect = [content, b""]
    with patch("urllib.request.urlopen", return_value=mock_resp):
        out = download_dataset("hopper-medium-v2", target_dir=tmp_path, expected_sha256=expected_hash)
        assert out.exists()


def test_format_valid_hdf5_without_required_fields_is_rejected(tmp_path):
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset("observations", data=np.zeros((5, 3)))
    payload = bio.getvalue()

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [payload, b""]
    mock_resp.headers.get.return_value = str(len(payload))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match="not a valid dataset"):
            download_dataset("hopper-medium-v2", target_dir=tmp_path)

    assert not (tmp_path / "hopper_medium-v2.hdf5").exists()
    assert not (tmp_path / "hopper_medium-v2.hdf5.tmp").exists()


def test_cached_schema_invalid_file_triggers_redownload(tmp_path):
    dest = tmp_path / "hopper_medium-v2.hdf5"
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset("observations", data=np.zeros((5, 3)))
    dest.write_bytes(bio.getvalue())

    content = make_dummy_hdf5_bytes()
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [content, b""]
    mock_resp.headers.get.return_value = str(len(content))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out = download_dataset("hopper-medium-v2", target_dir=tmp_path)
        assert out.read_bytes() == content
        mock_urlopen.assert_called_once()


def test_download_provenance_records_source_hash(tmp_path):
    content = make_dummy_hdf5_bytes()
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [content, b""]
    mock_resp.headers.get.return_value = str(len(content))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    provenance = tmp_path / "provenance.json"
    with patch("urllib.request.urlopen", return_value=mock_resp):
        download_dataset(
            "hopper-medium-v2", target_dir=tmp_path, provenance_path=provenance
        )

    import json
    payload = json.loads(provenance.read_text())
    import hashlib
    assert payload["source_sha256"] == hashlib.sha256(content).hexdigest()
    assert payload["dataset_info"]["transitions"] == 10
    assert payload["dataset_info"]["state_dim"] == 4
    assert payload["checksum_source"] == "computed_sha256"


def test_download_dataset_corrupted_cached_file_triggers_redownload(tmp_path):
    dest = tmp_path / "hopper_medium-v2.hdf5"
    dest.write_bytes(b"corrupted not hdf5")

    content = make_dummy_hdf5_bytes()
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [content, b""]
    mock_resp.headers.get.return_value = str(len(content))
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    # Force is False, but since dest is corrupted, it should re-download!
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out = download_dataset("hopper-medium-v2", target_dir=tmp_path, force=False)
        assert out == dest
        assert out.read_bytes() == content
        mock_urlopen.assert_called_once()

