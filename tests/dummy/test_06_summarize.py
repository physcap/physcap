"""physcap.summarize: success rule and the SR / OI / Time table.

    PYTHONPATH=. python -m pytest tests/dummy/test_06_summarize.py -q
"""

import json

from physcap.summarize import load_reports, summarize, trial_succeeded


def _report(trial, measurements, *, found=None, correct=None, seconds=100.0, label="physcap"):
    report = {
        "trial": trial,
        "task": "demo",
        "run_label": label,
        "totals": {"measurements": measurements},
        "target_object": {"found": bool(found)},
        "times": {"trial_total_s": seconds},
    }
    if correct is not None or found is None:
        report["decision"] = {"correct": correct}
    return report


def test_success_rule():
    assert trial_succeeded(_report(1, 2, found=True))
    assert not trial_succeeded(_report(1, 2, found=False))
    # A decision block decides, whatever was measured.
    assert trial_succeeded(_report(1, 2, found=False, correct=True))
    assert not trial_succeeded(_report(1, 2, found=True, correct=False))
    # A task that never declared counts as a failure, not as missing.
    assert not trial_succeeded(_report(1, 2, correct=None))
    # A person watching the real robot can override.
    assert not trial_succeeded({**_report(1, 2, found=True), "success": False})


def test_oi_and_time_average_over_successes_only(tmp_path):
    reports = [
        _report(1, 2, found=True, seconds=100),
        _report(2, 4, found=True, seconds=200),
        _report(3, 9, found=False, seconds=900),
    ]
    for r in reports:
        folder = tmp_path / f"trial_{r['trial']:02d}"
        folder.mkdir()
        (folder / "measurement_report.json").write_text(json.dumps(r))

    [row] = summarize(load_reports(tmp_path))
    assert row["trials"] == 3
    assert abs(row["SR"] - 2 / 3) < 1e-9
    assert row["OI"].startswith("3.00")
    assert row["Time"].startswith("150.00")
