"""
test_03_automation.py

Auto mode: the dummy env resolves the agent's free-text object names onto
ground-truth keys and advances the scene by itself, so a trial needs no human
at the keyboard (which is what makes --trials N / num_workers > 1 usable).

Resolution order is alias table -> LLM fallback -> hard failure. It never
guesses: a wrong mass would silently corrupt the run.

The alias file is COPIED to a temp dir first, so your real
configs/identify_empty_can/aliases.json is never modified.

Tests:
  1. Alias hit resolves with the resolver pointed at a dead port (no LLM call)
  2. Punctuation / capitalization / filler words normalize to the same key
  3. A name matching two objects is a miss, not a coin flip
  4. A resolved name is cached back into the alias file and reused offline
  5. The scene follows the declared can; weighing the empty can is recorded, not shown
  6. An unresolvable name aborts the trial instead of returning 0.0 kg
  7. (--live only) the real LLM fallback resolves and caches

Usage:
    uv run python tests/dummy/test_03_automation.py
    PHYSCAP_TEST_LIVE=1 uv run python tests/dummy/test_03_automation.py   # needs the proxy
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TASK_DIR = REPO_ROOT / "configs/identify_empty_can"
SCENES = TASK_DIR / "scenes.json"
DEAD_PORT = "http://127.0.0.1:9/chat/completions"  # discard port: any call fails fast

# Not argparse: this module is also imported by pytest, whose own CLI args
# would otherwise land in sys.argv and fail parsing here. --live is opt-in and
# off by default, so an env var is enough.
class _Args:
    live = os.environ.get("PHYSCAP_TEST_LIVE") == "1"
    resolver_url = os.environ.get(
        "PHYSCAP_TEST_RESOLVER_URL", "http://127.0.0.1:8110/chat/completions"
    )


args = _Args()

os.environ["PHYSCAP_GT_MASS_PATH"] = str(TASK_DIR / "gt-mass.json")

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


def build(resolver_model="google/gemini-2.5-flash-lite", url=DEAD_PORT, aliases_path=None):
    return PiperDummyLowLevel(
        scenes_path=str(SCENES),
        scenario="tray_right",
        aliases_path=aliases_path or str(aliases_copy),
        auto=True,
        empty_mass_threshold=0.1,
        resolver_model=resolver_model,
        resolver_server_url=url,
    )


print("=== Test 03: Auto mode (alias resolution + scene flipping) ===\n")
print(f"alias file (temp copy): {aliases_copy}")

# ── 1. Alias hit, no LLM ────────────────────────────────────────────────────
print("\n[1] Alias hit -- resolver points at a dead port, so any LLM call would fail")
env = build()
keys = env._gt_mass.keys()
check("'blue monster can' -> 'blue can'", env.resolve_object_key("blue monster can", keys) == "blue can")
check("'red coca-cola can' -> 'red can'", env.resolve_object_key("red coca-cola can", keys) == "red can")

# ── 2. Normalization ────────────────────────────────────────────────────────
print("\n[2] Normalization -- case, punctuation and filler words don't matter")
variants = ["The Blue Monster Can!", "blue monster can", "  the BLUE   monster-can  "]
resolved = [env.resolve_object_key(v, keys) for v in variants]
for v, r in zip(variants, resolved):
    print(f"    {v!r:34s} -> {r!r}")
check("all spellings resolve to the same key", set(resolved) == {"blue can"})

# ── 3. Ambiguity is a miss ──────────────────────────────────────────────────
print("\n[3] 'monster can' matches BOTH the blue and black cans")
print("    The alias layer must decline rather than pick one. (In production it")
print("    then falls through to the LLM; here the resolver is off, so it raises.)")
env_nollm = build(resolver_model=None)
check("alias layer alone does not resolve it", env_nollm._match_alias("monster can", list(keys)) is None)
try:
    got = env_nollm.resolve_object_key("monster can", keys)
    check("ambiguous name refused", False, f"wrongly resolved to {got!r}")
except RuntimeError:
    check("ambiguous name refused", True, "raised instead of guessing")

# ── 4. Cache write, offline ─────────────────────────────────────────────────
print("\n[4] A resolved name is remembered; the config's alias file is never rewritten")
original_aliases = aliases_copy.read_text()
cache_path = aliases_copy.parent / "learned_aliases.json"
os.environ["PHYSCAP_ALIAS_CACHE"] = str(cache_path)
env_cache = build()
env_cache._llm_resolve = lambda object_name, keys: "green can"  # stand in for the LLM
novel = "The Sparkling Citrus Beverage"
key = env_cache.resolve_object_key(novel, keys)
os.environ.pop("PHYSCAP_ALIAS_CACHE")
check("stubbed resolver answer used", key == "green can")
check("config alias file untouched", aliases_copy.read_text() == original_aliases)
learned = json.loads(cache_path.read_text())
check("learned alias saved to PHYSCAP_ALIAS_CACHE", "sparkling citrus beverage" in learned["green can"])

env_replay = build(resolver_model=None, aliases_path=str(cache_path))  # LLM fallback disabled
check("a later run resolves from the saved table alone", env_replay.resolve_object_key(novel, keys) == "green can")

# ── 5. Scene photos ─────────────────────────────────────────────────────────
print("\n[5] The scene follows the declared can (one end photo per can)")
env_scene = PiperDummyLowLevel(
    scenes_path=str(SCENES),
    scenario="tray_right",
    aliases_path=str(aliases_copy),
    auto=True,
    empty_mass_threshold=0.1,
    target_key="red can",  # as in configs/identify_empty_can/dummy.yaml
    decision_label="EMPTY CAN",
    resolver_model=None,
    resolver_server_url=DEAD_PORT,
)
api = DummyPhysXApi(env_scene)
env_scene.reset()
print(f"    start: {env_scene._current_key!r}")

w = api.get_mass("blue monster can")  # 0.388 kg -> not empty
check("weighing a full can keeps the start photo", env_scene._current_key == "initial", f"{w} kg")

w = api.get_mass("red coca-cola can")  # 0.015 kg -> empty
check("weighing the empty can keeps the start photo", env_scene._current_key == "initial", f"{w} kg")
check("but the report records it was found on the 2nd measurement",
      env_scene.measurement_report()["target_object"]["measurement_index"] == 2)

env_scene.on_code_output("EMPTY CAN: red coca-cola can\n")
check("declaring it shows the red can on the tray", env_scene._current_key == "final:red can",
      str(env_scene._current_key))
api.get_mass("green sprite can")
check("a later measurement does not change the photo", env_scene._current_key == "final:red can")

env_scene.reset()
check("reset() restores 'initial' for the next trial", env_scene._current_key == "initial" and not env_scene._target_found)

# ── 6. Fail loudly ──────────────────────────────────────────────────────────
print("\n[6] An unresolvable name must abort, NOT return a bogus 0.0 kg")
env_strict = build(resolver_model=None)
api_strict = DummyPhysXApi(env_strict)
try:
    value = api_strict.get_mass("quantum flux capacitor")
    check("unresolvable name raises", False, f"returned {value} instead")
except RuntimeError as exc:
    check("unresolvable name raises", "could not resolve" in str(exc))

# ── 7. Live resolver (opt-in) ───────────────────────────────────────────────
print("\n[7] Real LLM fallback")
if args.live:
    print(f"    --live: calling the resolver at {args.resolver_url}")
    env_live = build(url=args.resolver_url)
    phrase = "the fizzy drink in the emerald green container"
    try:
        live_key = env_live.resolve_object_key(phrase, keys)
        cached = json.loads(aliases_copy.read_text())
        print(f"    resolved {phrase!r} -> {live_key!r}")
        check("live resolver returned a valid key", live_key in keys)
        check(
            "live answer was cached",
            any("fizzy drink" in a for a in cached.get(live_key, [])),
        )
    except RuntimeError as exc:
        # UNKNOWN is a legitimate answer -- the resolver only sees text aliases.
        check("live resolver failed cleanly (no guess)", "could not resolve" in str(exc), str(exc)[:60])
else:
    print("    skipped -- pass --live to run it (costs an API call, needs the proxy up)")

# ── Summary ─────────────────────────────────────────────────────────────────
os.environ.pop("PHYSCAP_GT_MASS_PATH", None)
shutil.rmtree(tmp_dir, ignore_errors=True)
print("\n" + "=" * 60)
print("TEST 03 SUMMARY")
print("=" * 60)
for name, ok in results.items():
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
failed = [n for n, ok in results.items() if not ok]
print("=" * 60)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
print(f"repo alias file untouched: {TASK_DIR / 'aliases.json'}")


def test_checks_passed():
    assert not failed, f"{len(failed)} check(s) failed: {failed}"


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
