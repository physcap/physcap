"""Measurement bookkeeping and decision grading, shared by the dummy and real Piper envs.

Two things live here:

``MeasurementTracker``
    An ordered log of the physical-property measurements made during one trial.

``MeasurementMixin``
    The env-side surface built on that log: free-text object name -> ground-truth
    key resolution, the ``on_property_measured`` / ``on_code_output`` hooks the
    trial loop calls, and the ``measurement_report()`` it writes out.

A dummy trial and a hardware trial of the same task mix this in and so produce
the same report schema, comparable directly.
"""

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

_NONWORD_RE = re.compile(r"[^a-z0-9]+")
_ARTICLE_RE = re.compile(r"\b(?:the|a|an|of|on|at|is|its)\b")

# The canonical line a "choose one object" task asks the agent to print. The
# label is per-task -- "FIRST ITEM" for a grocery-packing decision, "Ripe
# avocado confirmed" for a ripeness task -- because a phrase that does not
# name what the task asks for reads as boilerplate and gets paraphrased away
# by the agent. Kept to horizontal whitespace so `^`/`$` anchor to real line
# boundaries under re.M, and case-sensitive so a passing mention in prose is
# not read as a commitment.
_DEFAULT_DECISION_LABEL = "FIRST ITEM"


def _decision_re(label: str) -> re.Pattern[str]:
    return re.compile(rf"^[ \t]*{re.escape(label)}:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


_DECISION_RE = _decision_re(_DEFAULT_DECISION_LABEL)

_DEFAULT_RESOLVER_MODEL = "google/gemini-2.5-flash-lite"


def _default_resolver_server_url() -> str:
    """The LLM proxy the alias resolver falls back to when nothing overrides it.

    ``PHYSCAP_RESOLVER_SERVER_URL`` names the resolver's own endpoint. Absent
    that, ``PHYSCAP_SERVER_URL`` (set by ``physcap/run.py`` from ``--server-url``)
    points the resolver at whatever proxy the run is already using. Only if
    neither is set does this fall back to the project's standard local proxy
    port.
    """
    return (
        os.environ.get("PHYSCAP_RESOLVER_SERVER_URL")
        or os.environ.get("PHYSCAP_SERVER_URL")
        or "http://127.0.0.1:8110/chat/completions"
    )


def _normalize(name: str) -> str:
    """Lowercase, drop punctuation and filler words, collapse whitespace."""
    text = _NONWORD_RE.sub(" ", str(name).lower())
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _strip_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop `_`-prefixed metadata keys (e.g. the `_units` note in a gt-*.json)."""
    return {k: v for k, v in raw.items() if not k.startswith("_")}


class MeasurementTracker:
    """Ordered log of the physical-property measurements made during one trial.

    How many objects the agent measured, and in what order, is the efficiency
    metric these trials exist to compare, and nothing else in the pipeline
    records it.

    Events are keyed by the *resolved* ground-truth key rather than the agent's
    free-text name, so measuring one object twice under two different phrasings
    is visible as a repeat instead of looking like two distinct objects.
    """

    SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Drop all events and restart the clock. Called on every env reset.

        The env is built once and reused for every trial, so without this a
        trial's report would inherit the previous trial's measurements.
        """
        self._events: list[dict[str, Any]] = []
        self._start = time.time()
        self._target_index: int | None = None
        self._target_found_key: str | None = None
        self._decision: dict[str, Any] | None = None

    def record(
        self,
        property_type: str,
        key: str | None,
        value: float,
        object_name: str | None = None,
        unit: str = "",
        turn: int | None = None,
    ) -> int:
        """Append one measurement and return its 1-based index."""
        index = len(self._events) + 1
        self._events.append(
            {
                "index": index,
                "turn": turn,
                "property": property_type,
                "object_name": object_name,
                # None in interactive mode with no ground-truth table loaded.
                "key": key,
                "value": value,
                "unit": unit,
                "t_since_trial_start_s": round(time.time() - self._start, 3),
            }
        )
        return index

    def mark_target_found(self, key: str | None) -> None:
        """Record which measurement identified the target object (empty can, ripe fruit, ...)."""
        if self._target_index is None:
            self._target_index = len(self._events)
            self._target_found_key = key

    @property
    def has_decision(self) -> bool:
        return self._decision is not None

    @property
    def decision(self) -> dict[str, Any] | None:
        return dict(self._decision) if self._decision is not None else None

    def record_decision(
        self,
        declared: str,
        key: str | None,
        correct: bool | None,
        turn: int | None = None,
    ) -> None:
        """Record the object the agent committed to, and how it was reached.

        Latches on the first declaration, like :meth:`mark_target_found` -- a
        task that asks for one choice gets one answer, and a later restatement
        must not overwrite it or reset the measurement count that preceded it.

        ``measurements_before`` is the efficiency half of the metric: how much
        physical probing the agent spent before committing.
        """
        if self._decision is not None:
            return
        self._decision = {
            "declared": declared,
            "key": key,
            "resolved": key is not None,
            "correct": correct,
            "measurements_before": len(self._events),
            "turn": turn,
            "t_since_trial_start_s": round(time.time() - self._start, 3),
        }

    def report(self) -> dict[str, Any]:
        """Build the JSON-serialisable per-trial report."""
        by_property: dict[str, int] = {}
        for event in self._events:
            by_property[event["property"]] = by_property.get(event["property"], 0) + 1

        # Fall back to the free-text name so an unkeyed measurement still counts
        # as its own object rather than collapsing every None into one.
        order = [e["key"] for e in self._events]
        identities = [
            e["key"] if e["key"] is not None else f"?{e['object_name']}" for e in self._events
        ]
        distinct = len(set(identities))
        # A repeat is the same property measured on the same object twice.
        # Keying this on the object alone would count weighing a cube and then
        # probing its stiffness as redundant, when for tasks that trade the two
        # properties off against each other that is exactly the intended plan.
        probes = {(e["property"], ident) for e, ident in zip(self._events, identities)}

        return {
            "schema_version": self.SCHEMA_VERSION,
            "totals": {
                "measurements": len(self._events),
                "by_property": by_property,
                "distinct_objects": distinct,
                "distinct_probes": len(probes),
                "repeat_measurements": len(self._events) - len(probes),
            },
            "target_object": {
                "found": self._target_index is not None,
                "key": self._target_found_key,
                "measurement_index": self._target_index,
            },
            "order": order,
            "events": list(self._events),
        }


class MeasurementMixin:
    """Measurement logging, name resolution and decision grading for a low-level env.

    Mix in ahead of ``BaseEnv``, call :meth:`_init_measurement` from ``__init__``
    and :meth:`_reset_measurement` from ``reset()``. The env supplies the
    ground-truth tables as ``self._gt_mass`` / ``self._gt_stiffness``; this
    mixin only reads them.
    """

    def _init_measurement(
        self,
        aliases_path: str | None = None,
        target_key: str | None = None,
        resolver_model: str | None = _DEFAULT_RESOLVER_MODEL,
        resolver_server_url: str | None = None,
        resolver_api_key: str | None = None,
        decision_label: str = _DEFAULT_DECISION_LABEL,
    ) -> None:
        self._tracker = MeasurementTracker()
        # The canonical line this task's prompt asks for. Must stay in lockstep
        # with the `prompt:` block -- a label the prompt never asks for grades
        # every trial as "never declared".
        self._decision_label = decision_label
        self._decision_re = _decision_re(decision_label)
        # Set per-turn by PiperEnv.step when available; events carry it as
        # `turn` so measurement order can be read against the multi-turn loop.
        self.current_turn: int | None = None
        # Ground-truth answer for "choose one object" tasks. Only used to grade
        # a declaration -- it is never shown to the agent or used to steer it.
        self._target_key = str(target_key) if target_key is not None else None

        self._aliases_path = aliases_path
        self._aliases: dict[str, list[str]] = {}
        if aliases_path:
            self._aliases = {str(k): list(v) for k, v in _load_json(aliases_path).items()}
        self._resolver_model = resolver_model
        self._resolver_server_url = resolver_server_url or _default_resolver_server_url()
        self._resolver_api_key = resolver_api_key or self._discover_api_key()

    def _reset_measurement(self) -> None:
        """Clear the log for a new trial. One env instance serves every trial."""
        self._tracker.clear()
        self.current_turn = None

    @property
    def _tag(self) -> str:
        """Log prefix, so messages name whichever env actually mixed this in."""
        return type(self).__name__

    # ---- construction helpers ----------------------------------------------
    @staticmethod
    def _load_gt_file(path: str | None) -> dict[str, float] | None:
        """Load a gt-*.json table, dropping its `_`-prefixed metadata keys."""
        if not path:
            return None
        return {k: float(v) for k, v in _strip_metadata(_load_json(path)).items()}

    @staticmethod
    def _discover_api_key() -> str | None:
        for var in ("PHYSCAP_RESOLVER_API_KEY", "OPENROUTER_API_KEY"):
            if os.environ.get(var):
                return os.environ[var]
        key_file = Path(".openrouterkey")
        if key_file.is_file():
            for line in key_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
        return None

    # ---- name -> ground-truth key resolution -------------------------------
    def resolve_object_key(self, object_name: str, key_universe: Iterable[str]) -> str:
        """Map a free-text object name onto a ground-truth key.

        Alias table first, then a single LLM call as fallback (whose answer is
        remembered for the rest of the run). Raises if neither resolves it -- we
        never guess, because a wrong mass silently corrupts the trial.
        """
        keys = [str(k) for k in key_universe]
        normalized = _normalize(object_name)

        hit = self._match_alias(normalized, keys)
        if hit is not None:
            return hit

        if self._resolver_model:
            resolved = self._llm_resolve(object_name, keys)
            if resolved in keys:
                self._cache_alias(resolved, normalized)
                return resolved

        raise RuntimeError(
            f"{self._tag}: could not resolve object name '{object_name}' "
            f"(normalized: '{normalized}') to any of the ground-truth keys {keys}. "
            f"Add an alias for it in '{self._aliases_path}'."
        )

    def _match_alias(self, normalized: str, keys: list[str]) -> str | None:
        """Exact match, else unambiguous containment. Ambiguity counts as a miss."""
        candidates: set[str] = set()
        for key, aliases in self._aliases.items():
            if key not in keys:
                continue
            for alias in aliases:
                alias_norm = _normalize(alias)
                if not alias_norm:
                    continue
                if alias_norm == normalized:
                    return key
                if alias_norm in normalized or normalized in alias_norm:
                    candidates.add(key)
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _llm_resolve(self, object_name: str, keys: list[str]) -> str | None:
        from capx.llm.client import ModelQueryArgs, query_model

        table = "\n".join(
            f"  {key}: {', '.join(self._aliases.get(key, [])) or '(no aliases yet)'}"
            for key in keys
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You map an object description onto one of a fixed set of object IDs. "
                    "Answer with the ID alone, or the single word UNKNOWN if none clearly match."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Object IDs and their known aliases:\n{table}\n\n"
                    f'Description: "{object_name}"\n\n'
                    f"Reply with exactly one of {keys}, or UNKNOWN."
                ),
            },
        ]
        try:
            out = query_model(
                ModelQueryArgs(
                    model=self._resolver_model,
                    server_url=self._resolver_server_url,
                    api_key=self._resolver_api_key,
                    # Reasoning models spend part of the budget before emitting
                    # content, so a tiny cap truncates the answer to nothing.
                    temperature=0.0,
                    max_tokens=256,
                ),
                prompt,
            )
        except Exception as exc:
            print(f"[{self._tag}] Alias resolver call failed: {exc}")
            return None
        answer = str(out.get("content") or "").strip().strip("\"'.")
        print(f"[{self._tag}] Resolver mapped '{object_name}' -> '{answer}'")
        return answer if answer in keys else None

    def _cache_alias(self, key: str, normalized: str) -> None:
        """Remember a newly resolved alias for the rest of this run.

        The alias table in ``configs/`` is never rewritten. To keep what the
        resolver learns, set ``PHYSCAP_ALIAS_CACHE`` to a file path: the full
        table (config aliases plus learned ones) is saved there, and can be
        passed back as ``aliases_path`` later.
        """
        self._aliases.setdefault(key, [])
        if normalized in self._aliases[key]:
            return
        self._aliases[key].append(normalized)
        cache = os.environ.get("PHYSCAP_ALIAS_CACHE")
        if not cache:
            return
        target = Path(cache)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=target.parent, delete=False, encoding="utf-8"
            ) as tmp:
                json.dump(self._aliases, tmp, indent=2)
                tmp.write("\n")
                tmp_path = tmp.name
            os.replace(tmp_path, target)
            print(f"[{self._tag}] Cached alias '{normalized}' -> '{key}' in {target}")
        except OSError as exc:
            print(f"[{self._tag}] Warning: could not cache alias to {target}: {exc}")

    # ---- measurement hook --------------------------------------------------
    def on_property_measured(
        self,
        property_type: str,
        key: str | None,
        value: float,
        object_name: str | None = None,
        unit: str = "",
    ) -> None:
        """Log one measurement. Subclasses may extend for scene side effects."""
        self._tracker.record(
            property_type, key, value, object_name=object_name, unit=unit,
            turn=self.current_turn,
        )

    # ---- decision hook -----------------------------------------------------
    def _decision_key_universe(self) -> list[str]:
        """Keys a declared choice may resolve to, best source first.

        Ground truth is preferred because it is what the aliases are written
        against; the alias table is the fallback for a task that configures no
        ground truth at all.
        """
        for source in (
            getattr(self, "_gt_mass", None),
            getattr(self, "_gt_stiffness", None),
            self._aliases,
        ):
            if source:
                return [str(k) for k in source]
        return []

    def on_code_output(self, stdout: str) -> None:
        """Record a ``<decision_label>: <name>`` declaration printed by generated code.

        Tasks that ask the agent to *choose* rather than to search leave no
        reliable trace in the observation, so a canonical printed line is the
        only channel the code-as-policy execution model gives us. PiperEnv.step
        feeds stdout here after every step.

        This never raises. Unlike a measurement -- where a wrong value silently
        corrupts the trial and so must fail loudly -- an unparsed or unresolved
        declaration only costs us the grade for that trial, and aborting would
        throw away the measurement data we did collect.
        """
        if self._tracker.has_decision:
            return
        matches = self._decision_re.findall(stdout or "")
        if not matches:
            return
        # Last wins: generated code may print a running hypothesis before it
        # commits, and the final line is the one it acted on.
        declared = matches[-1].strip().strip("\"'`*.")
        if not declared:
            return

        # Under the multi-turn reasoning pipeline the agent refers to objects
        # by canonical id ("cube_3"), which means nothing to the alias table.
        # Swap the id for its scene description first -- descriptions already
        # resolve, since that is what generated code passes to the measurement
        # APIs.
        resolvable = self._expand_canonical_id(declared)

        key: str | None = None
        try:
            key = self.resolve_object_key(resolvable, self._decision_key_universe())
        except Exception as exc:
            print(f"[{self._tag}] Could not resolve declared choice '{declared}': {exc}")

        correct = None if self._target_key is None else (key == self._target_key)
        self._tracker.record_decision(declared, key, correct, turn=self.current_turn)
        verdict = "" if correct is None else (" -- correct" if correct else " -- incorrect")
        print(
            f"[{self._tag}] Declared {self._decision_label.lower()}: "
            f"'{declared}' -> key '{key}'{verdict}."
        )
        self._on_decision(key)

    def _on_decision(self, key: str | None) -> None:
        """Side effect of a graded declaration. No-op unless a subclass needs one."""
        return None

    def _expand_canonical_id(self, declared: str) -> str:
        """Return the scene description for a canonical object id, else `declared`.

        The labels are regenerated per run by the VLM, so this reads the live
        mapping rather than a hand-maintained table: `cube_3` is not guaranteed
        to be the same object in the next run, and a static alias would then be
        confidently wrong instead of merely unresolved.
        """
        knowledge_dir = getattr(self, "knowledge_dir", None)
        if not knowledge_dir:
            return declared
        try:
            # Imported lazily so the env stays usable without the agents
            # package, as with the alias resolver's LLM client.
            from physcap.agents.scene_knowledge import load_object_labels

            labels = load_object_labels(Path(knowledge_dir))
        except Exception as exc:
            print(f"[{self._tag}] Could not read object labels: {exc}")
            return declared

        target = _normalize(declared)
        for obj in labels:
            if not isinstance(obj, dict):
                continue
            if _normalize(obj.get("id", "")) == target:
                description = str(obj.get("description") or "").strip()
                if description:
                    print(f"[{self._tag}] Declared id '{declared}' -> scene label '{description}'")
                    return description
        return declared

    # ---- report ------------------------------------------------------------
    def measurement_report(self) -> dict[str, Any]:
        """Per-trial measurement report. Read by the trial loop at trial end.

        Success is read as ``decision.correct`` when a ``"decision"`` block is
        present (a "choose one object" task), else ``target_object.found`` (a
        search task).
        """
        report = self._tracker.report()

        # Only decision tasks carry this block, so search tasks (and their
        # rollups) are untouched. A configured target means the block is always
        # present, with nulls when the agent never declared -- otherwise a trial
        # that failed to answer would silently drop out of the accuracy
        # denominator instead of counting against it.
        if self._target_key is not None or self._tracker.has_decision:
            report["target_key"] = self._target_key
            # Provenance: which canonical line this run was actually grading, so
            # a report showing declared=null can be told apart from a report
            # produced by a prompt/label mismatch.
            report["decision_label"] = self._decision_label
            report["decision"] = self._tracker.decision or {
                "declared": None,
                "key": None,
                "resolved": False,
                "correct": None,
                "measurements_before": None,
                "turn": None,
                "t_since_trial_start_s": None,
            }
        return report


__all__ = ["MeasurementTracker", "MeasurementMixin"]
