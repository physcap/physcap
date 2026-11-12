"""Summarise PhysCaP runs the way the paper reports them.

    uv run python -m physcap.summarize outputs/identify_empty_can

Walks the folder for ``measurement_report.json`` files (one per trial) and
prints, per method:

    SR    success rate over all trials
    OI    objects interacted: physical measurements before the task ended,
          counting a repeated measurement of the same object again
    Time  trial wall-clock time in seconds

As in the paper, OI and Time are averaged over successful trials only.

A trial succeeds when its declared answer is correct (tasks that print a
``<label>: <object>`` line, such as ``FIRST ITEM: green cube``), or otherwise
when it measured the target object. On the real robot, success also depends on
the physical placement, which only a person watching can judge; add a
``"success": true/false`` field to a trial's report to override it.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def trial_succeeded(report: dict[str, Any]) -> bool:
    if "success" in report:
        return bool(report["success"])
    if report.get("decision"):
        return report["decision"].get("correct") is True
    return bool((report.get("target_object") or {}).get("found"))


def load_reports(root: str | Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(Path(root).rglob("measurement_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[summarize] skipping {path}: {exc}")
            continue
        report["_path"] = str(path)
        reports.append(report)
    return reports


def _mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def summarize(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in reports:
        groups[(r.get("task") or "?", r.get("run_label") or r.get("method") or "?")].append(r)

    rows = []
    for (task, label), items in sorted(groups.items()):
        ok = [r for r in items if trial_succeeded(r)]
        rows.append({
            "task": task,
            "method": label,
            "trials": len(items),
            "SR": len(ok) / len(items),
            "OI": _mean_std([r["totals"]["measurements"] for r in ok]),
            "Time": _mean_std([(r.get("times") or {}).get("trial_total_s") for r in ok
                               if (r.get("times") or {}).get("trial_total_s") is not None]),
        })
    return rows


def print_run_summary(root: str | Path) -> None:
    rows = summarize(load_reports(root))
    if not rows:
        return
    tw = max(22, max(len(str(row["task"])) for row in rows) + 2)
    header = f"{'task':<{tw}}{'method':<20}{'n':>4}{'SR':>7}   {'OI':<16}{'Time (s)':<18}"
    print("\n" + header + "\n" + "-" * len(header))
    for row in rows:
        print(f"{row['task']:<{tw}}{row['method']:<20}{row['trials']:>4}{row['SR']:>7.0%}   "
              f"{row['OI']:<16}{row['Time']:<18}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m physcap.summarize <output folder>")
    print_run_summary(sys.argv[1])
