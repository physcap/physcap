"""Per-trial timing log.

Every trial writes ``time_log.txt`` next to its other artifacts. It splits the
wall-clock time of a trial into:

    A  environment setup     reset and first camera frame
    B  model calls           scene knowledge, Planner, Prioritizer, coding agent
    C  robot execution       each ``env.step``; on the real robot also split into
                             perception, arm motion, gripper, and measurement time
    D  saving artifacts

The paper's "Time" column is the trial total. The robot-execution subtotal
(C2 + C3 + C4) is printed as well, because it excludes model latency.
"""

from __future__ import annotations

import datetime
from pathlib import Path


class TrialTimer:
    """Collects ``(category, seconds, turn)`` events for one trial."""

    def __init__(self) -> None:
        self._events: list[tuple[str, float, int | None]] = []

    def record(self, category: str, duration: float, turn: int | None = None) -> None:
        self._events.append((category, duration, turn))

    def total(self, category: str, turn: int | None = None) -> float:
        return sum(d for c, d, t in self._events if c == category and (turn is None or t == turn))

    def count(self, category: str, turn: int | None = None) -> int:
        return sum(1 for c, _, t in self._events if c == category and (turn is None or t == turn))

    def turns(self) -> list[int]:
        return sorted({t for _, _, t in self._events if t is not None})


_MODEL_CALLS = (
    ("B1_scene_knowledge_init", "scene knowledge init"),
    ("B2_planner", "Planner"),
    ("B3_prioritizer", "Prioritizer"),
    ("B4_codegen", "coding agent"),
    ("B5_knowledge_update", "scene knowledge update"),
)
_ROBOT_PARTS = (
    ("C1_perception_molmo", "perception (Molmo)"),
    ("C2_robot_motion", "arm motion"),
    ("C3_gripper_op", "gripper"),
    ("C4_mass_measurement", "mass measurement"),
    ("C4_stiffness_measurement", "stiffness measurement"),
)
_ROBOT_EXECUTION = ("C2_robot_motion", "C3_gripper_op", "C4_mass_measurement", "C4_stiffness_measurement")


def write_time_log(
    timer: TrialTimer,
    output_path: Path,
    trial_total: float,
    webcam_info: dict | None = None,
) -> dict[str, float]:
    """Write ``time_log.txt`` and return the headline numbers."""
    width = 56

    def row(label: str, seconds: float, indent: int = 0, note: str = "") -> str:
        pad = "  " * indent
        return f"{pad}{label:<{38 - 2 * indent}}{seconds:>8.2f} s" + (f"   {note}" if note else "")

    lines = ["=" * width, "Trial timing", "=" * width, ""]

    setup = timer.total("A1_env_reset") + timer.total("A2_initial_scene")
    lines += ["[A] Environment setup", row("reset + first frame", setup, 1), ""]

    model_total = sum(timer.total(key) for key, _ in _MODEL_CALLS)
    n_calls = sum(timer.count(key) for key, _ in _MODEL_CALLS)
    lines.append("[B] Model calls")
    lines.append(row("scene knowledge init", timer.total("B1_scene_knowledge_init"), 1))
    for turn in timer.turns():
        per_turn = [(label, timer.total(key, turn)) for key, label in _MODEL_CALLS[1:]]
        if any(seconds for _, seconds in per_turn):
            lines.append(f"  -- turn {turn}")
            lines += [row(label, seconds, 2) for label, seconds in per_turn if seconds]
    lines += [row("subtotal", model_total, 1, f"({n_calls} call(s))"), ""]

    step_total = timer.total("C_total")
    lines.append("[C] Robot execution (env.step wall time)")
    for turn in timer.turns():
        if not timer.count("C_total", turn):
            continue
        lines.append(row(f"turn {turn}", timer.total("C_total", turn), 1))
        for key, label in _ROBOT_PARTS:
            n = timer.count(key, turn)
            if n:
                lines.append(row(label, timer.total(key, turn), 2, f"({n} call(s))"))
    lines += [row("subtotal", step_total, 1, f"({timer.count('C_total')} step(s))"), ""]

    saving = timer.total("D1_artifact_save")
    lines += ["[D] Saving artifacts", row("subtotal", saving, 1), ""]

    robot_execution = sum(timer.total(key) for key in _ROBOT_EXECUTION)
    lines += [
        "=" * width,
        row("Trial total", trial_total),
        row("Model calls (B)", model_total),
        row("Robot execution (C2+C3+C4)", robot_execution),
        row("Unaccounted", max(0.0, trial_total - setup - model_total - step_total - saving)),
        "=" * width,
    ]

    if webcam_info:
        def clock(t: float) -> str:
            return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]

        lines += [
            "",
            "[F] Webcam recording",
            row("start offset", webcam_info["start_offset_s"], 1, f"(at {clock(webcam_info['start_wall'])})"),
            row("duration", webcam_info["duration"], 1, f"({webcam_info['frames']} frames)"),
            f"  video: {webcam_info['video_path']}",
        ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return {
        "trial_total_s": trial_total,
        "model_calls_s": model_total,
        "robot_execution_s": robot_execution,
    }
