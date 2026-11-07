import json
import subprocess
import sys

import pytest
import numpy as np


def invoke(*args):

    return subprocess.run([sys.executable, '-m', 'udt', *map(str, args)],
                          capture_output=True, text=True)


def test_cli_demo_audit_prepare_roundtrip(tmp_path):
    source = tmp_path / 'demo.npz'
    assert invoke('demo-data', '--output', source).returncode == 0
    audit_path = tmp_path / 'audit.json'
    result = invoke('audit', source, '--horizon', 3, '--starts-at-reset',
                    '--timeout-semantics', 'horizon', '--output', audit_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(audit_path.read_text())
    assert report['gate_a'] == 'REVIEW_REQUIRED'
    assert report['provenance']['dataset_id'] == 'unspecified'
    assert report['rtg_eligible_episodes'] == 4
    out = tmp_path / 'prepared'
    result = invoke('prepare', source, '--horizon', 3, '--starts-at-reset',
                    '--timeout-semantics', 'horizon', '--output', out)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / 'manifest.json').read_text())
    assert len(manifest['provenance']['source_sha256']) == 64
    assert manifest['provenance']['command'][0] == 'prepare'
    with np.load(out / 'dataset.npz', allow_pickle=False) as data:
        assert len(data['returns_to_go']) == 12
        assert set(data['episode_ids'][data['is_validation']]).isdisjoint(
            data['episode_ids'][~data['is_validation']])
    assert invoke('prepare', source, '--horizon', 3, '--starts-at-reset',
                  '--timeout-semantics', 'horizon', '--output', out).returncode != 0


def test_cli_rejects_pickle_and_bad_data_with_actionable_message(tmp_path):
    path = tmp_path / 'old.pkl'
    path.write_bytes(b'not a pickle')
    result = invoke('audit', path, '--horizon', 3, '--output', tmp_path / 'a.json')
    assert result.returncode != 0
    assert 'NPZ or HDF5' in result.stderr
    assert not (tmp_path / 'a.json').exists()
