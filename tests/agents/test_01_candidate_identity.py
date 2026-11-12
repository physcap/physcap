"""
test_01_candidate_identity.py

Exploration-candidate identity. The Planner is told to use the canonical object
id as a candidate's `name`, so `name` alone identifies the OBJECT, not the
action. Keying the candidate pool on it would collapse every probe of one
object into a single entry: Pack Grocery Bag asks for mass AND stiffness
per cube, the Planner correctly emits 8 candidates for 4 cubes, and only 4
would survive -- every stiffness probe silently dropped.

Identity is the (object, action) pair via `candidate_key`, and scores-mode
prioritization restores the winner's full candidate before handing it to
codegen. Without that second half a fixed pool changes nothing: the scored
entry carries only {name, action, reason, priority_score}, so codegen would
receive an object id and no instruction, and improvise its own action.

No network, no hardware, no model calls -- every function under test is pure.

Tests:
  1. candidate_key separates probes, absorbs duplicates, tolerates odd input
  2. Missing / unspecific action falls back to the description
  3. Merge keeps both probes of an object and still drops true duplicates
  4. Executing one probe leaves its sibling pending
  5. Scores mode restores description/parameters onto the winner
  6. The Planner schema advertises `action` so a route can enforce it
  7. Scores-mode validation accepts probes that share a name
  8. Probes actually performed retire their candidates, however they were batched
  9. Accumulated knowledge reaches the Prioritizer's prompt, and is optional
 10. PhysCaP-joint follows THIS turn's ordering, not the frozen pool order

Usage:
    uv run python tests/agents/test_01_candidate_identity.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from physcap.agents.cycle import (  # noqa: E402
    filter_pending_candidates,
    merge_new_candidates,
    probed_candidate_keys,
    select_from_fresh_ordering,
    select_scored_winner,
)
from physcap.agents.planner import (  # noqa: E402
    ACTION_VOCAB,
    TASK_EXECUTE_NAME,
    candidate_key,
    exploration_candidate_schema,
)
from physcap.agents.prioritizer import (  # noqa: E402
    _build_original_candidate_map,
    _validate_scored_candidates,
    build_prioritizer_user_text,
)

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def cand(name: str, action: str, desc: str) -> dict:
    return {
        "name": name,
        "action": action,
        "description": desc,
        "parameters": {"object_name": "string"},
        "expected_info": f"{action} of {name}",
        "estimated_cost": "medium",
    }


# The 8 candidates the planner actually produced for Pack Grocery Bag.
COLOURS = ["blue", "red", "green", "yellow"]
EIGHT = [
    cand(f"cube_{i}", action, f"Measure the {prop} of the {colour} cube")
    for i, colour in enumerate(COLOURS)
    for action, prop in (("measure_mass", "mass"), ("measure_stiffness", "stiffness"))
]

print("=== Test 01: Exploration-candidate identity ===\n")
print("Scenario: 4 cubes x 2 properties = 8 candidates, sharing 4 names\n")

# ── 1. candidate_key ────────────────────────────────────────────────────────
print("[1] candidate_key is the (object, action) pair, not the name")
k_mass = candidate_key(EIGHT[0])
k_stiff = candidate_key(EIGHT[1])
print(f"    cube_0 mass       -> {k_mass!r}")
print(f"    cube_0 stiffness -> {k_stiff!r}")

check("both probes share a name", EIGHT[0]["name"] == EIGHT[1]["name"] == "cube_0")
check("but get different keys", k_mass != k_stiff)
check("all 8 candidates are distinct", len({candidate_key(c) for c in EIGHT}) == 8,
      str(len({candidate_key(c) for c in EIGHT})))
check("an identical candidate re-keys the same",
      candidate_key(cand("cube_0", "measure_mass", "Measure the mass of the blue cube"))
      == k_mass)
check("separator drift is absorbed",
      candidate_key({"name": "cube-0", "action": "Measure_Mass", "description": "x"})
      == candidate_key({"name": "cube_0", "action": "measure mass", "description": "x"}))
check("bare string normalizes", candidate_key("cube_0") == "cube 0", candidate_key("cube_0"))
check("task-execute sentinel keeps its bare name",
      candidate_key({"name": TASK_EXECUTE_NAME, "description": "Execute task directly"})
      == "original task execute",
      candidate_key({"name": TASK_EXECUTE_NAME, "description": "Execute task directly"}))
check("None is survivable", candidate_key(None) == "")

# ── 2. Fallback ─────────────────────────────────────────────────────────────
print("\n[2] Missing or unspecific action falls back to the description")
print("    (routes that cannot enforce response_json_schema may omit `action`;")
print("     with no discriminator at all, every probe would collide again)")
no_action_w = {"name": "cube_0", "description": "Measure the mass of the blue cube"}
no_action_s = {"name": "cube_0", "description": "Measure the stiffness of the blue cube"}
check("two actionless probes still separate",
      candidate_key(no_action_w) != candidate_key(no_action_s))
check("'other' also falls back to description",
      candidate_key({"name": "cube_0", "action": "other", "description": "shake it"})
      != candidate_key({"name": "cube_0", "action": "other", "description": "tilt it"}))
check("no action and no description degrades to the bare name",
      candidate_key({"name": "cube_0"}) == "cube 0")

# ── 3. Merge ────────────────────────────────────────────────────────────────
print("\n[3] The merge keeps every probe")
pool: list[dict] = []
added = merge_new_candidates(pool, EIGHT)
print(f"    planner emitted 8 -> pool holds {len(pool)}")
check("all 8 candidates enter the pool", len(pool) == 8, str(len(pool)))
check("merge reports what it added", added == 8, str(added))
check("stiffness probes survived",
      sum(1 for c in pool if c["action"] == "measure_stiffness") == 4)

re_added = merge_new_candidates(pool, EIGHT)
check("re-proposing the same 8 adds nothing", re_added == 0 and len(pool) == 8,
      f"added={re_added}, pool={len(pool)}")
new_one = merge_new_candidates(pool, [cand("cube_0", "relocate", "Move the blue cube aside")])
check("a genuinely new probe is still admitted", new_one == 1 and len(pool) == 9)

# ── 4. Executed filter ──────────────────────────────────────────────────────
print("\n[4] Executing one probe must not retire its sibling")
pool8 = []
merge_new_candidates(pool8, EIGHT)
executed = [candidate_key(EIGHT[2])]          # cube_1 mass
print(f"    executed: {executed}")
pending = filter_pending_candidates(pool8, executed, [])
pending_keys = {candidate_key(c) for c in pending}

check("the executed probe is gone", candidate_key(EIGHT[2]) not in pending_keys)
check("its stiffness sibling is still pending", candidate_key(EIGHT[3]) in pending_keys)
check("7 of 8 remain", len(pending) == 7, str(len(pending)))
check("failed plans are filtered too",
      len(filter_pending_candidates(pool8, executed, [candidate_key(EIGHT[3])])) == 6)
check("a bare object name does not retire a keyed probe",
      len(filter_pending_candidates(pool8, ["cube_1"], [])) == 8)

# ── 5. Scores mode ──────────────────────────────────────────────────────────
print("\n[5] Scores mode must restore the winner's instruction")
scored = [
    {"name": "cube_1", "action": "measure_mass", "reason": "closest", "priority_score": 10},
    {"name": "cube_0", "action": "measure_stiffness", "reason": "far", "priority_score": 3},
]
selected = select_scored_winner(scored, EIGHT)
print(f"    selected: {selected.get('name')} / {selected.get('action')}")
print(f"    description: {selected.get('description')!r}")

check("highest score wins", selected["name"] == "cube_1" and selected["priority_score"] == 10)
check("description restored (was '' before the fix)",
      selected.get("description") == "Measure the mass of the red cube",
      repr(selected.get("description")))
check("parameters restored", selected.get("parameters") == {"object_name": "string"})
check("expected_info restored", bool(selected.get("expected_info")))
check("prioritizer's reason kept", selected.get("reason") == "closest")

stiff_winner = select_scored_winner(
    [{"name": "cube_1", "action": "measure_stiffness", "reason": "r", "priority_score": 9}],
    EIGHT,
)
check("the sibling probe resolves to ITS own candidate",
      stiff_winner["description"] == "Measure the stiffness of the red cube",
      repr(stiff_winner["description"]))
check("an unknown winner degrades to the bare scored entry",
      select_scored_winner(
          [{"name": "ghost", "action": "inspect", "reason": "r", "priority_score": 5}], EIGHT
      ) == {"name": "ghost", "action": "inspect", "reason": "r", "priority_score": 5})
check("no scored candidates -> None", select_scored_winner([], EIGHT) is None)

# ── 6. Schema and the prioritizer map ───────────────────────────────────────
print("\n[6] The schema advertises `action`, and the restore map is key-aware")
schema = exploration_candidate_schema()
check("`action` is a schema property", "action" in schema["properties"])
check("`action` is required", "action" in schema["required"])
check("`action` is a closed enum", schema["properties"]["action"].get("enum") == list(ACTION_VOCAB),
      str(schema["properties"]["action"].get("enum")))
check("`action` is ordered right after `name`",
      schema["propertyOrdering"][:2] == ["name", "action"])
check("measure_mass replaces measure_weight in the vocabulary",
      "measure_mass" in ACTION_VOCAB and "measure_weight" not in ACTION_VOCAB)

omap = _build_original_candidate_map(EIGHT)
check("the map holds all 8 by key",
      all(candidate_key(c) in omap for c in EIGHT))
check("a keyed lookup returns the right sibling",
      omap[candidate_key(EIGHT[1])]["description"] == "Measure the stiffness of the blue cube")
check("an ambiguous bare name resolves to the first candidate shown",
      omap["cube_0"]["description"] == "Measure the mass of the blue cube",
      omap["cube_0"]["description"])

# ── 7. Scores-mode validation ───────────────────────────────────────────────
print("\n[7] Validation must accept 8 entries over 4 names")
omap8 = _build_original_candidate_map(EIGHT)


def as_scored(entries):
    return {"scored_candidates": entries}


good = [
    {"name": c["name"], "action": c["action"], "reason": "r", "priority_score": 5}
    for c in EIGHT
]
check("8 correctly-keyed entries validate",
      _validate_scored_candidates(as_scored(good), EIGHT, omap8) is True)

dupe = [dict(e) for e in good]
dupe[1] = dict(dupe[0])
check("a genuine duplicate (name+action) is still rejected",
      _validate_scored_candidates(as_scored(dupe), EIGHT, omap8) is False)

check("a short list is still rejected",
      _validate_scored_candidates(as_scored(good[:4]), EIGHT, omap8) is False)

bad_score = [dict(e) for e in good]
bad_score[0]["priority_score"] = 99
check("an out-of-range score is still rejected",
      _validate_scored_candidates(as_scored(bad_score), EIGHT, omap8) is False)

fallback = [
    {"name": c.get("name", ""), "action": c.get("action", ""),
     "reason": "Fallback", "priority_score": 5}
    for c in EIGHT
]
fb_winner = select_scored_winner(fallback, EIGHT)
check("fallback entries carry `action`", all(e.get("action") for e in fallback))
check("so a fallback winner still restores its description",
      fb_winner.get("description") == "Measure the mass of the blue cube",
      repr(fb_winner.get("description")))

actionless = [{"name": "cube_1", "reason": "r", "priority_score": 7}]
check("even an action-less winner gets SOME instruction",
      bool(select_scored_winner(actionless, EIGHT).get("description")),
      repr(select_scored_winner(actionless, EIGHT).get("description")))

# ── 8. Probe accounting ─────────────────────────────────────────────────────
print("\n[8] Probes actually performed retire their own candidates")
print("    (codegen may batch mass+stiffness in one step; retiring only the")
print("     selected candidate would leave the other to be offered again)")
LABELS = [
    {"id": f"cube_{i}", "description": f"the {colour} cube"}
    for i, colour in enumerate(COLOURS)
] + [{"id": "bag_0", "description": "the brown paper bag on the right"}]

batched = [
    {"object_name": "the blue cube", "property_type": "mass", "turn": 1},
    {"object_name": "the blue cube", "property_type": "stiffness", "turn": 1},
]
keys = probed_candidate_keys(batched, LABELS)
print(f"    -> {sorted(keys)}")
check("both batched probes are retired", keys == {
    candidate_key(EIGHT[0]), candidate_key(EIGHT[1])}, str(sorted(keys)))

pool8b = []
merge_new_candidates(pool8b, EIGHT)
check("and the pool shrinks by 2",
      len(filter_pending_candidates(pool8b, sorted(keys), [])) == 6,
      str(len(filter_pending_candidates(pool8b, sorted(keys), []))))

check("a canonical id also matches",
      probed_candidate_keys(
          [{"object_name": "cube_2", "property_type": "mass"}], LABELS)
      == {candidate_key(EIGHT[4])})
check("an unknown object retires nothing",
      probed_candidate_keys(
          [{"object_name": "a purple hovercraft", "property_type": "mass"}], LABELS)
      == set())
check("a malformed record retires nothing",
      probed_candidate_keys([{"object_name": "the blue cube"}, None, "x"], LABELS) == set())
check("no labels retires nothing", probed_candidate_keys(batched, []) == set())

# ── 9. Knowledge reaches the prioritizer ────────────────────────────────────
print("\n[9] Scene knowledge is carried into the prioritizer prompt")
print("    (without it the ranking sees only static properties, so it cannot")
print("     demote a candidate an earlier measurement already ruled out)")

KNOWLEDGE = (
    "The yellow cube fourth from the left has a mass of 0.509 kg.\n"
    "The red cube second from the left has a mass of 0.105 kg."
)
with_k = build_prioritizer_user_text(EIGHT, scene_knowledge=KNOWLEDGE)
without_k = build_prioritizer_user_text(EIGHT)

check("the measured values appear verbatim", "0.509 kg" in with_k and "0.105 kg" in with_k)
check("under a labelled heading", "## Accumulated Scene Knowledge" in with_k)
check("with guidance on how to use it", "can no longer" in with_k)
check("candidates are still present", '"measure_stiffness"' in with_k)

check("omitting knowledge adds no heading", "Scene Knowledge" not in without_k)
check("and leaves the rest unchanged",
      without_k == build_prioritizer_user_text(EIGHT, scene_knowledge=None))
check("empty string is treated as absent",
      "Scene Knowledge" not in build_prioritizer_user_text(EIGHT, scene_knowledge=""))
check("whitespace-only is treated as absent",
      "Scene Knowledge" not in
      build_prioritizer_user_text(EIGHT, scene_knowledge="   \n  "))

check("the image note is independent of knowledge",
      "scene image is attached" in build_prioritizer_user_text(EIGHT, has_image=True)
      and "scene image is attached" not in with_k)

import inspect as _inspect  # noqa: E402
from physcap.agents.prioritizer import prioritizer as _prioritizer  # noqa: E402
_sig = _inspect.signature(_prioritizer)
check("prioritizer() exposes scene_knowledge", "scene_knowledge" in _sig.parameters)
check("and it defaults to None, so calls without it still work",
      _sig.parameters["scene_knowledge"].default is None)
check("the property-prior parameter is gone", "priors" not in _sig.parameters)

_cycle_src = (REPO_ROOT / "physcap/agents/cycle.py").read_text(encoding="utf-8")
check("cycle.py passes scene knowledge to the prioritizer",
      "scene_knowledge=_load_scene_knowledge(output_dir)" in _cycle_src)

_policy = (REPO_ROOT / "prompts/prioritizer.txt").read_text(encoding="utf-8")
check("the ladder lives in prompts/prioritizer.txt",
      "Rung 1" in _policy and "Rung 2" in _policy)
check("and travel is explicitly ruled out as a criterion",
      "NOT ranking criteria" in _policy)
check("the cost table uses measure_mass, not measure_weight",
      "measure_mass" in _policy and "measure_weight" not in _policy)

# ── 10. Fresh ordering (PhysCaP-joint) ──────────────────────────────────────
print("\n[10] The joint Planner's re-ranking must actually be followed")
print("     (merge_new_candidates skips a re-proposed candidate rather than")
print("      re-appending it, so pool position is frozen at first appearance --")
print("      selecting pending[0] would follow turn 1's ranking for the whole trial)")

pool10: list[dict] = []
merge_new_candidates(pool10, EIGHT)          # pool is in EIGHT's order

# Turn 2: the planner re-ranks, putting cube_2's mass first.
fresh = [EIGHT[4], EIGHT[0], EIGHT[2]]
picked = select_from_fresh_ordering(fresh, pool10)
print(f"     pool head = {pool10[0]['name']}/{pool10[0]['action']}, "
      f"fresh head = {fresh[0]['name']}/{fresh[0]['action']} -> picked "
      f"{picked['name']}/{picked['action']}")

check("this turn's ordering wins over insertion order",
      candidate_key(picked) == candidate_key(EIGHT[4]), candidate_key(picked))
check("which is NOT the pool head", candidate_key(picked) != candidate_key(pool10[0]))

# An executed candidate must be skipped in favour of the next fresh one.
pending10 = filter_pending_candidates(pool10, [candidate_key(EIGHT[4])], [])
skipped = select_from_fresh_ordering(fresh, pending10)
check("an executed candidate is skipped for the next fresh one",
      candidate_key(skipped) == candidate_key(EIGHT[0]), candidate_key(skipped))

# The merge must backfill fields a thin re-emission dropped.
thin = {"name": "cube_2", "action": "measure_mass",
        "description": "Weigh the green one"}          # no parameters/expected_info
merged = select_from_fresh_ordering([thin], pool10)
check("fresh wording wins", merged["description"] == "Weigh the green one",
      merged["description"])
check("pooled fields are backfilled",
      merged.get("parameters") == {"object_name": "string"} and bool(merged.get("expected_info")),
      str(merged.get("parameters")))
print("     (codegen reads description/parameters off this; a stripped candidate")
print("      would leave the coding agent to invent the action itself)")

# Fallback: nothing fresh is pending.
done_all = [candidate_key(c) for c in EIGHT[:6]]
leftovers = filter_pending_candidates(pool10, done_all, [])
fb = select_from_fresh_ordering([EIGHT[0]], leftovers)   # EIGHT[0] already executed
check("fallback returns the most recently merged pending",
      candidate_key(fb) == candidate_key(leftovers[-1]), candidate_key(fb))
check("and not the insertion-ordered head",
      candidate_key(fb) != candidate_key(leftovers[0]))

# Degenerate inputs must not raise.
check("empty fresh list falls back", select_from_fresh_ordering([], pool10) is not None)
check("None fresh list falls back", select_from_fresh_ordering(None, pool10) is not None)
check("no pending at all -> None", select_from_fresh_ordering(fresh, []) is None)
check("both empty -> None", select_from_fresh_ordering([], []) is None)

# The artifact must record what was followed, not the pool order.
# Scope the assertion to the follow_fresh_ordering branch rather than the whole
# file, so this cannot pass because some unrelated line happens to mention
# new_candidates.
_joint_branch = _cycle_src.split("elif follow_fresh_ordering:")[1].split("\n    else:")[0]
check("joint mode writes the fresh ordering to prioritized_candidates.json",
      "json.dumps(new_candidates" in _joint_branch
      and 'prioritized_json_path = turn_dir / "prioritized_candidates.json"' in _joint_branch)
check("joint mode does not point that path at the pending pool",
      "prioritized_json_path = pending_candidates_path" not in _joint_branch)
check("plain planner mode still records the pool order",
      "prioritized_json_path = pending_candidates_path" in _cycle_src)

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 01 SUMMARY")
print("=" * 60)
failed = [n for n, ok in results.items() if not ok]
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")


def test_checks_passed():
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
