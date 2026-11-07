import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from udt.download import D4RL_URLS, download_dataset


def test_download_dataset_unknown_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        download_dataset("non-existent-v2")


def test_download_dataset_mocked_http(tmp_path):
    fake_content = b"fake hdf5 content data stream"

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
    dest.write_bytes(b"already exists")

    # Should not invoke urlopen because file exists and force=False
    with patch("urllib.request.urlopen") as mock_urlopen:
        out_path = download_dataset("hopper-medium-v2", target_dir=tmp_path, force=False)
        assert out_path == dest
        mock_urlopen.assert_not_called()

    # With force=True, it should re-download
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"new content", b""]
    mock_resp.headers.get.return_value = "11"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        out_path = download_dataset("hopper-medium-v2", target_dir=tmp_path, force=True)
        assert out_path.read_bytes() == b"new content"
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
