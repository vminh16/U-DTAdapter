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
