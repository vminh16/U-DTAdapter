import pytest
from udt.readiness import check_pilot_readiness
from test_readiness import _make_artifacts, _run


def test_duplicate_bootstrap_seeds_fail(tmp_path):
    """Duplicated bootstrap seeds across ensemble heads must fail distinct-stream check."""
    artifacts = _make_artifacts(tmp_path)
    # Head 0 has seed 0, Head 1 also given seed 0
    artifacts["dynamics_payload"]["metrics"]["models"][0]["bootstrap"]["bootstrap_seed"] = 0
    artifacts["dynamics_payload"]["metrics"]["models"][1]["bootstrap"]["bootstrap_seed"] = 0
    artifacts["dynamics_payload"]["bootstrap"] = [{"bootstrap_seed": 0}, {"bootstrap_seed": 0}]

    report = _run(artifacts)
    assert report["checks"]["bootstrap"]["status"] == "fail"
    problems = report["checks"]["bootstrap"]["evidence"]["problems"]
    assert any("distinct" in p.lower() or "duplicate" in p.lower() for p in problems)


def test_bootstrap_metadata_inconsistency_fails(tmp_path):
    """Inconsistency between top-level checkpoint bootstrap metadata and metric records fails."""
    artifacts = _make_artifacts(tmp_path)
    # Top-level claims seed 999, metrics record seed 1
    artifacts["dynamics_payload"]["bootstrap"][1]["bootstrap_seed"] = 999
    artifacts["dynamics_payload"]["metrics"]["models"][1]["bootstrap"]["bootstrap_seed"] = 1

    report = _run(artifacts)
    assert report["checks"]["bootstrap"]["status"] == "fail"
    problems = report["checks"]["bootstrap"]["evidence"]["problems"]
    assert any("inconsistent" in p.lower() or "mismatch" in p.lower() or "metric record" in p.lower() for p in problems)


def test_distinct_bootstrap_seeds_pass(tmp_path):
    """Distinct seeds consistent across records and metadata pass cleanly."""
    artifacts = _make_artifacts(tmp_path)
    report = _run(artifacts)
    assert report["checks"]["bootstrap"]["status"] == "pass"
    assert report["checks"]["bootstrap"]["evidence"]["bootstrap_seeds"] == [0, 1]
