"""Fully decoupled Piper low-level env: no daemon, no hardware, no camera.

Serves a human-supplied photo as the camera observation instead of a live ZED
frame, so the reasoning/code-generation pipeline can run against a Piper task
with no physical arm, camera, or piper_daemon.py involved. Pairs with
DummyPhysXApi, which turns control calls into no-ops and physical property
measurements into either manual human input (interactive mode) or
ground-truth lookups (auto mode).

Measurement logging, object-name resolution and ``<decision_label>:`` decision
grading are shared with the real-hardware env and live in
``physcap.measurement``; only the scene-flip behaviour built on top of
them is dummy-specific.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.base import BaseEnv
from physcap.measurement import (
    _DEFAULT_DECISION_LABEL,
    _DEFAULT_RESOLVER_MODEL,
    _load_json,
    MeasurementMixin,
    MeasurementTracker,  # noqa: F401 -- re-exported for callers that import it from here
)


class PiperDummyLowLevel(MeasurementMixin, BaseEnv):
    """No-hardware Piper low-level env that serves a human-supplied photo.

    Photo sources (exactly one required):
      - ``scenes_path``: a JSON file. Two accepted shapes:
          * scenario pairs -- ``{"tray_right": {"initial": ..., "final": ...}}``.
            Pick one with ``scenario``. The frame starts on ``initial``.
            ``final`` may be either:
              - a path, for search tasks: the frame flips once the target
                object's mass is measured; or
              - a ``{gt-key: path}`` map, for tasks that ask the agent to
                *choose* one object: the frame flips to the photo matching
                whichever object the agent declares (see
                :meth:`on_code_output`), and the mass threshold is ignored.
          * flat -- ``{"0": path, "1": path}``. Selected by key, starting at
            ``initial_scene``.
      - ``initial_image_path``: a single fixed photo.

    In interactive mode (``auto=False``) a menu is printed once per turn so a
    human can swap the frame. In auto mode nothing is ever prompted: the frame
    advances on its own and physical properties are resolved from ground-truth
    files, which is what makes ``--trials N`` / ``num_workers > 1`` usable.

    Ground truth is read from ``gt_mass_path`` / ``gt_stiffness_path`` (JSON
    tables of object key -> value); the ``PHYSCAP_GT_MASS_PATH`` and
    ``PHYSCAP_GT_STIFFNESS_PATH`` environment variables override them.
    Object names produced by generated code are mapped onto those keys by
    :meth:`resolve_object_key`.
    """

    def __init__(
        self,
        initial_image_path: str | None = None,
        scenes_path: str | None = None,
        initial_scene: str = "0",
        scenario: str | None = None,
        auto: bool = False,
        aliases_path: str | None = None,
        gt_mass_path: str | None = None,
        gt_stiffness_path: str | None = None,
        empty_mass_threshold: float = 0.1,
        ripe_stiffness_threshold: float | None = None,
        target_key: str | None = None,
        decision_label: str = _DEFAULT_DECISION_LABEL,
        resolver_model: str | None = _DEFAULT_RESOLVER_MODEL,
        resolver_server_url: str | None = None,
        resolver_api_key: str | None = None,
        privileged: bool = False,
        enable_render: bool = True,
        viser_debug: bool = False,
    ) -> None:
        if (initial_image_path is None) == (scenes_path is None):
            raise ValueError(
                "PiperDummyLowLevel: pass exactly one of initial_image_path or scenes_path."
            )

        self._auto = auto
        self._empty_mass_threshold = float(empty_mass_threshold)
        self._ripe_stiffness_threshold = (
            float(ripe_stiffness_threshold) if ripe_stiffness_threshold is not None else None
        )
        self._target_found = False
        self.obs: dict[str, Any] = {}
        self._skip_next_prompt = False
        self.knowledge_dir: str | None = None

        # ---- photo source -------------------------------------------------
        self._scene_mode: str  # "pair" | "flat" | "single"
        self._scenarios: dict[str, dict[str, str]] = {}
        self._scenario: str | None = None
        self._scenes: dict[str, str] = {}
        self._current_key: str | None = None
        self._initial_image_path: str | None = None

        if scenes_path is not None:
            raw = _load_json(scenes_path)
            if raw and all(isinstance(v, dict) for v in raw.values()):
                self._scene_mode = "pair"
                self._scenarios = raw
                self._scenario = self._pick_scenario(scenario, scenes_path)
                states = self._scenarios[self._scenario]
                for state in ("initial", "final"):
                    if state not in states:
                        raise ValueError(
                            f"PiperDummyLowLevel: scenario '{self._scenario}' in '{scenes_path}' "
                            f"is missing the '{state}' photo."
                        )
                if not isinstance(states["initial"], str):
                    raise ValueError(
                        f"PiperDummyLowLevel: scenario '{self._scenario}' in '{scenes_path}' "
                        f"must give 'initial' as a single photo path."
                    )
                self._verify_files(states, scenes_path)
            else:
                self._scene_mode = "flat"
                self._scenes = raw
                self._verify_files(self._scenes, scenes_path)
                if initial_scene not in self._scenes:
                    raise ValueError(
                        f"PiperDummyLowLevel: initial_scene '{initial_scene}' not in "
                        f"'{scenes_path}'. Available keys: {sorted(self._scenes)}."
                    )
                self._initial_scene = initial_scene
        else:
            if not Path(initial_image_path).is_file():
                raise FileNotFoundError(
                    f"PiperDummyLowLevel: initial_image_path '{initial_image_path}' does not exist."
                )
            self._scene_mode = "single"
            self._initial_image_path = initial_image_path

        # ---- ground truth --------------------------------------------------
        # The environment variables override the config, e.g. to try another table.
        self._gt_mass = self._load_gt_file(os.environ.get("PHYSCAP_GT_MASS_PATH") or gt_mass_path)
        self._gt_stiffness = self._load_gt_file(
            os.environ.get("PHYSCAP_GT_STIFFNESS_PATH") or gt_stiffness_path
        )

        # ---- measurement log, name resolution, decision grading ------------
        self._init_measurement(
            aliases_path=aliases_path,
            target_key=target_key,
            decision_label=decision_label,
            resolver_model=resolver_model,
            resolver_server_url=resolver_server_url,
            resolver_api_key=resolver_api_key,
        )

        self._load_initial()

    # ---- construction helpers ---------------------------------------------
    def _pick_scenario(self, scenario: str | None, scenes_path: str) -> str:
        if scenario is not None:
            if scenario not in self._scenarios:
                raise ValueError(
                    f"PiperDummyLowLevel: scenario '{scenario}' not in '{scenes_path}'. "
                    f"Available: {sorted(self._scenarios)}."
                )
            return scenario
        if len(self._scenarios) == 1:
            return next(iter(self._scenarios))
        raise ValueError(
            f"PiperDummyLowLevel: '{scenes_path}' defines multiple scenarios "
            f"{sorted(self._scenarios)}; set `scenario` to choose one."
        )

    @staticmethod
    def _verify_files(mapping: dict[str, Any], scenes_path: str) -> None:
        """Existence-check every photo path, descending one level so that
        outcome-keyed ``final`` maps are checked too."""
        missing: list[str] = []
        for key, value in mapping.items():
            if isinstance(value, dict):
                missing += [f"{key}:{k}" for k, path in value.items() if not Path(path).is_file()]
            elif not Path(value).is_file():
                missing.append(key)
        if missing:
            raise FileNotFoundError(
                f"PiperDummyLowLevel: '{scenes_path}' references missing files for {missing}."
            )

    # ---- frames ------------------------------------------------------------
    def _final_frames(self) -> dict[str, str] | None:
        """Outcome-keyed final frames, or None when 'final' is a single photo.

        Doubles as the test for which flip mechanism this scenario uses: a map
        means the frame follows the agent's declared choice, a path means it
        follows the target-object mass threshold.
        """
        if self._scene_mode != "pair":
            return None
        final = self._scenarios[self._scenario]["final"]
        return final if isinstance(final, dict) else None

    def _selectable(self) -> dict[str, str]:
        """The frames a human may switch between this turn."""
        if self._scene_mode == "pair":
            options: dict[str, str] = {}
            for state, value in self._scenarios[self._scenario].items():
                if isinstance(value, dict):
                    options.update({f"{state}:{k}": path for k, path in value.items()})
                else:
                    options[state] = value
            return options
        return dict(self._scenes)

    def _load_initial(self) -> None:
        self._target_found = False
        if self._scene_mode == "pair":
            self._current_key = "initial"
            self._load_frame(self._scenarios[self._scenario]["initial"])
        elif self._scene_mode == "flat":
            self._current_key = self._initial_scene
            self._load_frame(self._scenes[self._initial_scene])
        else:
            self._current_key = self._initial_image_path
            self._load_frame(self._initial_image_path)

    def _load_frame(self, path: str) -> None:
        rgb = np.array(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)
        intrinsics = np.array(
            [[530.0, 0.0, w / 2.0], [0.0, 530.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self.obs = {
            "robot0_robotview": {
                "images": {"rgb": rgb, "depth": depth},
                "intrinsics": intrinsics,
                "pose_mat": np.eye(4, dtype=np.float32),
            }
        }

    # ---- measurement hook --------------------------------------------------
    def _threshold_hit(self, property_type: str, value: float) -> bool:
        """Whether this measurement, by itself, identifies the target object.

        Two independent thresholds so one env class serves both a "find the
        empty container" task (mass) and a "find the ripe fruit" task
        (stiffness): a search task configures whichever one its physical
        property is, and the other stays ``None`` and never fires.
        """
        if property_type == "mass" and self._empty_mass_threshold is not None:
            return float(value) < self._empty_mass_threshold
        if property_type == "stiffness" and self._ripe_stiffness_threshold is not None:
            return float(value) <= self._ripe_stiffness_threshold
        return False

    def on_property_measured(
        self,
        property_type: str,
        key: str | None,
        value: float,
        object_name: str | None = None,
        unit: str = "",
    ) -> None:
        """Log the measurement, then advance the scene if the object was the target.

        Logging happens first and unconditionally: every measurement belongs in
        the report, including measurements of other properties/objects that the
        scene-flip logic below deliberately ignores.
        """
        super().on_property_measured(
            property_type, key, value, object_name=object_name, unit=unit
        )

        if self._scene_mode != "pair" or self._target_found:
            return
        if self._final_frames() is not None:
            # Outcome-keyed finals: the frame follows the agent's declared
            # choice (on_code_output), not a threshold. Measuring a candidate
            # is not an answer, so the scene stays. The report still records
            # when the graded answer was first measured, if it crosses the
            # threshold (Identify Empty Can: the red can, below 0.1 kg).
            if (
                self._target_key is not None
                and key == self._target_key
                and self._threshold_hit(property_type, value)
            ):
                self._tracker.mark_target_found(key)
            return
        if self._threshold_hit(property_type, value):
            self._tracker.mark_target_found(key)
            if self._target_key is not None:
                # Graded on the declaration, so the scene waits for it (see
                # _on_decision). On the robot, measuring the can does not put it
                # on the tray; showing it on the tray here would tell the Planner
                # the task is finished before the agent has declared anything.
                return
            self._target_found = True
            self._current_key = "final"
            self._load_frame(self._scenarios[self._scenario]["final"])
            print(
                f"[PiperDummyLowLevel] Target object found (key '{key}', "
                f"{property_type}={value}) -- scene advanced to '{self._scenario}/final'."
            )

    # ---- decision hook -----------------------------------------------------
    def _decision_key_universe(self) -> list[str]:
        """As the mixin, but the outcome-keyed final frames are also a source.

        A scenario may enumerate one photo per possible choice without
        configuring any ground truth, and those keys are then the only
        statement of what the agent is choosing between.
        """
        for source in (self._gt_mass, self._gt_stiffness, self._final_frames(), self._aliases):
            if source:
                return [str(k) for k in source]
        return []

    def _on_decision(self, key: str | None) -> None:
        """Swap in the final photo for a declared choice, if the scenario has one.

        Two shapes of ``final``, two behaviours:
          - outcome-keyed (a ``{key: path}`` map): swap to whichever photo
            matches the declared key, right or wrong -- the agent gets to see
            the outcome of its own choice.
          - a single photo: this is a "find the target object" task with a
            decision label (e.g. "Ripe avocado confirmed: <id>"), and the
            photo is the "found it" scene. Flip to it only on a *correct*
            declaration, and only if a threshold measurement has not already
            found it -- whichever comes first wins, and a wrong declaration
            must not reveal the answer.
        """
        frames = self._final_frames()
        if frames is not None:
            path = frames.get(key) if key is not None else None
            if path is None:
                print(
                    f"[PiperDummyLowLevel] No final photo for '{key}'; "
                    f"keeping scene '{self._current_key}'."
                )
                return
            self._target_found = True
            self._current_key = f"final:{key}"
            self._load_frame(path)
            print(f"[PiperDummyLowLevel] Scene advanced to '{self._scenario}/final:{key}'.")
            return

        if self._target_found or self._target_key is None or key != self._target_key:
            return
        self._target_found = True
        self._tracker.mark_target_found(key)
        self._current_key = "final"
        self._load_frame(self._scenarios[self._scenario]["final"])
        print(
            f"[PiperDummyLowLevel] Correct declaration ('{key}') -- "
            f"scene advanced to '{self._scenario}/final'."
        )

    def measurement_report(self) -> dict[str, Any]:
        """Per-trial measurement report, plus which scene setup produced it."""
        report = super().measurement_report()
        report["scenario"] = self._scenario
        report["empty_mass_threshold"] = self._empty_mass_threshold
        report["ripe_stiffness_threshold"] = self._ripe_stiffness_threshold
        return report

    # ---- gym surface -------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._load_initial()
        self._skip_next_prompt = True
        # One env instance is reused for every trial, so the previous trial's
        # measurements must not leak into this one's report.
        self._reset_measurement()
        return self.obs, {}

    def get_observation(self) -> dict[str, Any]:
        if self._skip_next_prompt:
            self._skip_next_prompt = False
            return self.obs
        if self._auto:
            return self.obs

        if self._scene_mode == "single":
            new_path = input(
                f"[PiperDummyLowLevel] Current photo: '{self._current_key}'. "
                "Enter a new photo path to swap for this turn, or press Enter to keep it: "
            ).strip()
            if new_path:
                self._current_key = new_path
                self._load_frame(new_path)
            return self.obs

        options_map = self._selectable()
        print("Available scenes:")
        for key, path in options_map.items():
            print(f"  {key}: {path}")
        new_key = input(
            f"[PiperDummyLowLevel] Current scene: '{self._current_key}'. "
            "Enter a scene key to swap for this turn, or press Enter to keep it: "
        ).strip()
        if new_key:
            if new_key in options_map:
                self._current_key = new_key
                self._load_frame(options_map[new_key])
            else:
                print(
                    f"[PiperDummyLowLevel] Warning: '{new_key}' is not a known scene key. "
                    "Keeping the current scene."
                )
        return self.obs

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # Unused: CodeExecutionEnvBase executes generated code directly and never
        # calls low_level_env.step() for Piper's code-as-policy execution model.
        return self.obs, 0.0, False, False, {}

    def compute_reward(self) -> float:
        """Cannot be auto-computed without a real robot/scene; use human evaluation."""
        return 0.0

    def task_completed(self) -> bool:
        """Cannot be auto-detected without a real robot/scene; use human evaluation."""
        return False

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        rgb = self.obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
        if rgb is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return rgb


__all__ = ["MeasurementTracker", "PiperDummyLowLevel"]
