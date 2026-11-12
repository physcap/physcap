"""High-level PiPER task env: runs generated code and turn bookkeeping.

Sits on top of ``capx.envs.tasks.base.CodeExecutionEnvBase``, which does the
actual code execution. This adds the multi-turn bookkeeping the physics tasks
need on top: a turn counter shared with the low-level env's measurement log,
per-turn timing, and the scene-knowledge directory a decision task's canonical
object-id references are read against.
"""

from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling an AgileX PiPER 6-DOF robot arm to solve a puzzle.
"""


class PiperEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._turn = 0
        self.step_timings: list[tuple[int, float]] = []
        self.step_events: list[tuple[int, str, float]] = []
        self._knowledge_dir: str | None = None

    def reset(self, *, seed=None, options=None):
        self._turn = 0
        self.step_timings = []
        self.step_events = []
        return super().reset(seed=seed, options=options)

    def step(self, code: str):
        import time

        self._turn += 1
        if hasattr(self.low_level_env, "current_turn"):
            self.low_level_env.current_turn = self._turn

        _t0 = time.time()
        obs, reward, terminated, truncated, info = super().step(code)
        self.step_timings.append((self._turn, time.time() - _t0))

        # Drain the low-level env's per-step timing events (perception, arm
        # motion, gripper, measurement) into our own turn-tagged log, so a
        # trial's time breakdown survives across turns rather than being
        # overwritten by the next step.
        events = getattr(self.low_level_env, "_step_timing_events", None)
        if events:
            for category, seconds in events:
                self.step_events.append((self._turn, category, seconds))
            events.clear()

        on_code_output = getattr(self.low_level_env, "on_code_output", None)
        if callable(on_code_output):
            try:
                on_code_output(info.get("stdout", ""))
            except Exception as exc:
                print(f"[PiperEnv] Warning: on_code_output failed: {exc}")

        return obs, reward, terminated, truncated, info

    @property
    def knowledge_dir(self) -> str | None:
        return self._knowledge_dir

    @knowledge_dir.setter
    def knowledge_dir(self, value: str | None) -> None:
        self._knowledge_dir = value
        self.low_level_env.knowledge_dir = value


__all__ = ["PiperEnv"]
