"""
test_05_decision_report.py

Decision tasks. A search task asks the agent to SEARCH -- success is a
measured mass below a threshold, and the env can see it happen. Tasks like
Pack Grocery Bag instead ask the agent to CHOOSE, and a choice leaves no
trace in the observation: goto_pose is a no-op and nothing in the scene says
which cube was picked.

So the prompt requires one canonical printed line, `FIRST ITEM: <name>`, and the
trial loop feeds each step's stdout to PiperDummyLowLevel.on_code_output(). The
env resolves the name against the alias table, grades it against `target_key`,
and swaps in the final photo for that specific choice.

Unlike a measurement -- where a wrong value silently corrupts the trial and so
must fail loudly -- a declaration that cannot be parsed or resolved only costs
the grade for that trial, so the hook never raises.

The alias file is COPIED to a temp dir first, so your real
configs/pack_grocery_bag/aliases.json is never modified.

Tests:
  1. A declaration is parsed, resolved, graded, and flips the matching photo
  2. A wrong choice is graded wrong -- and still flips, to its own photo
  3. Outcome-keyed scenes never flip on a mass threshold
  4. The last declaration in stdout wins, and the first one latches
  5. Unparseable / unresolvable stdout never raises
  6. reset() clears the decision, so trial 2 does not inherit trial 1
  7. String-valued 'final' (the search-task shape) still flips on mass
  8. Report shape: decision block present iff it is a decision task
  9. A canonical object id from the reasoning pipeline resolves via object labels

Usage:
    uv run python tests/dummy/test_05_decision_report.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TASK_DIR = REPO_ROOT / "configs/pack_grocery_bag"
SCENES = TASK_DIR / "scenes.json"
CAN_DIR = REPO_ROOT / "configs/identify_empty_can"
DEAD_PORT = "http://127.0.0.1:9/chat/completions"  # discard port: any call fails fast

os.environ["PHYSCAP_GT_MASS_PATH"] = str(TASK_DIR / "gt-mass.json")
os.environ["PHYSCAP_GT_STIFFNESS_PATH"] = str(TASK_DIR / "gt-stiffness.json")

from physcap.envs.piper_dummy import PiperDummyLowLevel  # noqa: E402
from physcap.apis.dummy_api import DummyPhysXApi  # noqa: E402

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# Work on a throwaway copy so the repo's alias file is never touched.
tmp_dir = Path(tempfile.mkdtemp(prefix="piper_dummy_test_"))
aliases_copy = tmp_dir / "aliases.json"
shutil.copy(TASK_DIR / "aliases.json", aliases_copy)


def build(**overrides):
    kwargs = dict(
        scenes_path=str(SCENES),
        scenario="bag_right",
        aliases_path=str(aliases_copy),
        auto=True,
        target_key="green",
        # Inert for this scenario: outcome-keyed finals bypass the threshold.
        empty_mass_threshold=0.0,
        # Resolver disabled: every name used below is already an alias, so a
        # network call here would mean the alias layer silently regressed.
        resolver_model=None,
        resolver_server_url=DEAD_PORT,
    )
    kwargs.update(overrides)
    return PiperDummyLowLevel(**kwargs)


print("=== Test 05: Declared-choice grading ===\n")
print(f"alias file (temp copy): {aliases_copy}")
print("gt: yellow is heaviest (0.419) but soft (3); green is heavy (0.418) AND rigid (5)")
print("target_key = 'green' -- only mass AND stiffness together identify it\n")

# ── 1. The happy path ───────────────────────────────────────────────────────
print("[1] A correct declaration is resolved, graded, and flips its own photo")
env = build()
api = DummyPhysXApi(env)
for name in ["blue cube", "red cube", "green cube", "yellow cube"]:
    api.get_mass(name)
for name in ["yellow cube", "green cube"]:
    api.get_stiffness(name)

check("scene has not moved before the declaration", env._current_key == "initial",
      str(env._current_key))

env.current_turn = 4
env.on_code_output("THOUGHT: green is heavy and rigid\nFIRST ITEM: green cube\n")
report = env.measurement_report()
decision = report["decision"]
print(f"    decision = {decision}")

check("declared text preserved", decision["declared"] == "green cube")
check("resolved to the GT key", decision["key"] == "green", str(decision["key"]))
check("marked resolved", decision["resolved"] is True)
check("graded correct", decision["correct"] is True)
check("measurement spend recorded", decision["measurements_before"] == 6,
      str(decision["measurements_before"]))
check("turn recorded", decision["turn"] == 4)
check("scene flipped to the green outcome", env._current_key == "final:green",
      str(env._current_key))
check("target_key reported for analysis", report["target_key"] == "green")

# Mass + stiffness on one cube is the intended plan here, not redundancy.
check("mass+stiffness on one cube is not a repeat",
      report["totals"]["repeat_measurements"] == 0,
      str(report["totals"]["repeat_measurements"]))
check("distinct objects counted by key", report["totals"]["distinct_objects"] == 4)
check("distinct probes counted by (property, object)",
      report["totals"]["distinct_probes"] == 6)

# ── 2. A wrong choice ───────────────────────────────────────────────────────
print("\n[2] The heaviest-but-soft distractor is graded wrong -- and still flips")
env2 = build()
DummyPhysXApi(env2).get_mass("yellow cube")
env2.on_code_output("FIRST ITEM: yellow cube\n")
d2 = env2.measurement_report()["decision"]

check("resolved to yellow", d2["key"] == "yellow", str(d2["key"]))
check("graded incorrect", d2["correct"] is False)
check("a wrong choice still advances the scene", env2._current_key == "final:yellow",
      str(env2._current_key))
print("    (a scene that froze on a wrong answer would leak correctness via")
print("     image differencing, so every declared choice gets its own photo)")

# ── 3. No threshold flipping ────────────────────────────────────────────────
print("\n[3] Outcome-keyed scenes ignore the mass threshold entirely")
env3 = build(empty_mass_threshold=0.5)   # would fire on red/blue (~0.105 kg)
api3 = DummyPhysXApi(env3)
api3.get_mass("red cube")
api3.get_mass("blue cube")

check("a light object did not flip the scene", env3._current_key == "initial",
      str(env3._current_key))
check("no empty object was marked",
      env3.measurement_report()["target_object"]["found"] is False)
print("    (measuring a light object is not an answer when the task is to choose)")

# ── 4. Last wins, first latches ─────────────────────────────────────────────
print("\n[4] The last line in one stdout wins; the first declaration latches")
env4 = build()
env4.on_code_output(
    "FIRST ITEM: blue cube\n"
    "wait, recomputing...\n"
    "FIRST ITEM: green cube\n"
)
check("last declaration in a block wins",
      env4.measurement_report()["decision"]["key"] == "green",
      str(env4.measurement_report()["decision"]["key"]))

env4.on_code_output("FIRST ITEM: red cube\n")
check("a later step cannot overwrite the committed choice",
      env4.measurement_report()["decision"]["key"] == "green")
check("and cannot move the scene again", env4._current_key == "final:green")

# ── 5. Bad input never raises ───────────────────────────────────────────────
print("\n[5] Unparseable or unresolvable stdout is survivable, not fatal")
env5 = build()
for stdout in ("", "no declaration here\n", "first item: lowercase does not match\n"):
    env5.on_code_output(stdout)
check("no declaration -> nothing recorded",
      env5.measurement_report()["decision"]["declared"] is None)
check("no declaration -> scene unmoved", env5._current_key == "initial")

env5.on_code_output("FIRST ITEM: a purple hovercraft\n")   # not in the alias table
d5 = env5.measurement_report()["decision"]
check("unresolvable name is still recorded verbatim",
      d5["declared"] == "a purple hovercraft", str(d5["declared"]))
check("unresolved -> null key", d5["key"] is None)
check("unresolved -> resolved False", d5["resolved"] is False)
check("unresolved -> graded incorrect, not unknown", d5["correct"] is False)
check("unresolved -> scene held", env5._current_key == "initial")
print("    (a failed grade must not throw away the measurement data alongside it)")

# ── 6. Reset ────────────────────────────────────────────────────────────────
print("\n[6] reset() clears the decision -- one env instance serves every trial")
env6 = build()
DummyPhysXApi(env6).get_mass("green cube")
env6.on_code_output("FIRST ITEM: green cube\n")
check("decision recorded before reset",
      env6.measurement_report()["decision"]["key"] == "green")

env6.reset()
r6 = env6.measurement_report()
check("decision cleared", r6["decision"]["declared"] is None)
check("correct cleared to null, not False", r6["decision"]["correct"] is None)
check("measurements cleared", r6["totals"]["measurements"] == 0)
check("scene restored to initial", env6._current_key == "initial")

# ── 7. The search-task shape still works ─────────────────────────────────
print("\n[7] String-valued 'final' -- search tasks flip on mass")
os.environ["PHYSCAP_GT_MASS_PATH"] = str(CAN_DIR / "gt-mass.json")
can_aliases = tmp_dir / "can-aliases.json"
shutil.copy(CAN_DIR / "aliases.json", can_aliases)
# A one-photo `final` (the search-task shape), built from the Empty Can photos.
single_scenes = tmp_dir / "single-final-scenes.json"
single_scenes.write_text(json.dumps({"single": {
    "initial": str(CAN_DIR / "scenes/0_initial.jpeg"),
    "final": str(CAN_DIR / "scenes/1_final_red.jpeg"),
}}))
env7 = PiperDummyLowLevel(
    scenes_path=str(single_scenes),
    scenario="single",
    aliases_path=str(can_aliases),
    auto=True,
    empty_mass_threshold=0.1,
    resolver_model=None,
    resolver_server_url=DEAD_PORT,
)
api7 = DummyPhysXApi(env7)
api7.get_mass("blue monster can")
check("a full can does not flip", env7._current_key == "initial")
api7.get_mass("red coca-cola can")   # the empty one, 0.015 kg
r7 = env7.measurement_report()
check("the empty can flips to 'final'", env7._current_key == "final",
      str(env7._current_key))
check("target_object still tracked", r7["target_object"]["measurement_index"] == 2)
check("search tasks carry no decision block", "decision" not in r7)
check("search tasks carry no target_key", "target_key" not in r7)

# The same scene graded on an "EMPTY CAN: <id>" line. The photo waits for the
# declaration: showing the can on the tray as soon as it is measured would tell
# the Planner the task is done before the agent has declared anything.
print("\n[7b] Graded on the declaration, the scene waits for it")
env7b = PiperDummyLowLevel(
    scenes_path=str(single_scenes),
    scenario="single",
    aliases_path=str(can_aliases),
    auto=True,
    empty_mass_threshold=0.1,
    target_key="red can",
    decision_label="EMPTY CAN",
    resolver_model=None,
    resolver_server_url=DEAD_PORT,
)
api7b = DummyPhysXApi(env7b)
api7b.get_mass("red coca-cola can")
r7b = env7b.measurement_report()
check("measuring the empty can does not flip", env7b._current_key == "initial",
      str(env7b._current_key))
check("but the find is still counted", r7b["target_object"]["measurement_index"] == 1)
check("no decision yet", r7b["decision"]["declared"] is None)
env7b.on_code_output("EMPTY CAN: red coca-cola can.\n")
r7b = env7b.measurement_report()
check("the declaration is graded correct", r7b["decision"]["correct"] is True)
check("and flips the scene", env7b._current_key == "final", str(env7b._current_key))

env7c = PiperDummyLowLevel(
    scenes_path=str(single_scenes),
    scenario="single",
    aliases_path=str(can_aliases),
    auto=True,
    empty_mass_threshold=0.1,
    target_key="red can",
    decision_label="EMPTY CAN",
    resolver_model=None,
    resolver_server_url=DEAD_PORT,
)
env7c.on_code_output("EMPTY CAN: blue monster can.\n")
check("a wrong declaration does not reveal the answer", env7c._current_key == "initial")
os.environ["PHYSCAP_GT_MASS_PATH"] = str(TASK_DIR / "gt-mass.json")

# ── 8. Shape ────────────────────────────────────────────────────────────────
print("\n[8] Report shape is stable and serializable")
check("schema_version present", report.get("schema_version") == 2)
check("whole report is JSON-serializable", isinstance(json.dumps(report), str))
for field in ("totals", "target_object", "order", "events", "decision", "target_key"):
    check(f"'{field}' present on a decision task", field in report)

# A configured target means the block is always there, even when the agent
# never answered -- otherwise a silent trial would drop out of the accuracy
# denominator instead of counting against it.
env8 = build()
check("decision block present even with no declaration",
      "decision" in env8.measurement_report())


class _PlainEnv:
    """Stands in for any non-dummy low-level env (franka, libero, piper_real)."""


check(
    "non-tracking env exposes no on_code_output hook",
    not callable(getattr(_PlainEnv(), "on_code_output", None)),
)

# ── 9. Canonical ids from the reasoning pipeline ────────────────────────────
print("\n[9] A declared canonical id resolves through the run's object labels")
print("    (the reasoning pipeline tells the agent to say 'cube_3'; the alias")
print("     table only knows colours, so a correct answer scored as wrong)")
reasoning_dir = tmp_dir / "reasoning"
reasoning_dir.mkdir(exist_ok=True)
(reasoning_dir / "object_labels.json").write_text(json.dumps([
    {"id": "cube_0", "description": "the blue cube on the far left"},
    {"id": "cube_2", "description": "the green cube third from the left"},
    {"id": "bag_0", "description": "the brown paper bag on the right"},
]), encoding="utf-8")

env9 = build()
env9.knowledge_dir = str(reasoning_dir)
env9.on_code_output("FIRST ITEM: cube_2\n")
d9 = env9.measurement_report()["decision"]
print(f"    decision = {d9}")

check("the id resolved to the right ground-truth key", d9["key"] == "green", str(d9["key"]))
check("graded correct", d9["correct"] is True)
check("the agent's own wording is preserved", d9["declared"] == "cube_2")
check("and the scene advanced", env9._current_key == "final:green", str(env9._current_key))

env9b = build()
env9b.knowledge_dir = str(reasoning_dir)
env9b.on_code_output("FIRST ITEM: cube_0\n")
check("a different id maps to its own object",
      env9b.measurement_report()["decision"]["key"] == "blue")
check("and is graded incorrect", env9b.measurement_report()["decision"]["correct"] is False)

env9c = build()
env9c.knowledge_dir = str(reasoning_dir)
env9c.on_code_output("FIRST ITEM: cube_9\n")
check("an id with no label still fails soft",
      env9c.measurement_report()["decision"]["resolved"] is False)

env9d = build()
env9d.knowledge_dir = str(tmp_dir / "does_not_exist")
env9d.on_code_output("FIRST ITEM: yellow cube\n")
check("a missing labels file does not break plain colour names",
      env9d.measurement_report()["decision"]["key"] == "yellow")

env9e = build()   # no knowledge_dir at all (--mode none)
env9e.on_code_output("FIRST ITEM: yellow cube\n")
check("and neither does having no reasoning dir",
      env9e.measurement_report()["decision"]["key"] == "yellow")

# ── Summary ─────────────────────────────────────────────────────────────────
os.environ.pop("PHYSCAP_GT_MASS_PATH", None)
shutil.rmtree(tmp_dir, ignore_errors=True)
os.environ.pop("PHYSCAP_GT_STIFFNESS_PATH", None)
print("\n" + "=" * 60)
print("TEST 05 SUMMARY")
print("=" * 60)
failed = [n for n, ok in results.items() if not ok]
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
print(f"repo alias file untouched: {TASK_DIR / 'aliases.json'}")


def test_checks_passed():
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
