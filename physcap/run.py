"""Run a PhysCaP experiment: one task, one method, N trials.

    uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap

``--method`` picks one of the five methods compared in the paper:

    cap                 CaP (CaP-Agent0): the coding agent alone, no measurement tools
    cap_physx           CaP + PhysX: adds get_mass / get_stiffness
    cap_physx_planner   CaP + PhysX + Planner: the Planner decides what to explore and when to stop
    physcap_joint       PhysCaP-joint: Planner and Prioritizer merged into one model call
    physcap             PhysCaP: Planner, then a separate Prioritizer ranks the candidates

Everything else (robot or dummy, scene, prompts, models) comes from the task
YAML and can be overridden on the command line. All CaP-X flags
(``capx.envs.launch.LaunchArgs``) are accepted too.

This reuses CaP-X's runner unchanged; it only swaps in ``physcap.trial.run_trial``
as the per-trial function.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time
import urllib.request
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tyro

import physcap  # noqa: F401  (registers the PhysCaP envs and APIs with CaP-X)
from capx.envs.configs.loader import DictLoader
from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config

from physcap.agents.scene_knowledge import AGENT_SERVER_URL_ENV
from physcap.prompt_files import PROMPTS_DIR, task_prompts


# method -> (reasoning mode, give the coding agent the measurement tools?)
METHODS: dict[str, tuple[str, bool]] = {
    "cap": ("none", False),
    "cap_physx": ("none", True),
    "cap_physx_planner": ("planner", True),
    "physcap_joint": ("joint", True),
    "physcap": ("planner_prioritizer", True),
}

# The same scene with the measurement tools removed, for the "cap" method.
WITHOUT_PHYSX = {
    "PiperPhysXApi": "PiperControlApi",
    "DummyPhysXApi": "DummyControlApi",
    "LiberoPhysXApi": "LiberoControlApi",
}


@dataclass
class PhysCaPArgs(LaunchArgs):
    """Arguments for ``python -m physcap.run`` (plus every CaP-X flag)."""

    method: Literal["cap", "cap_physx", "cap_physx_planner", "physcap_joint", "physcap"] = "physcap"
    """Which method from the paper to run."""

    planner_model: str | None = None
    """Model for the Planner (and scene knowledge). Defaults to --model."""

    prioritizer_model: str | None = None
    """Model for the Prioritizer. Defaults to --model."""

    agent_server_url: str | None = None
    """OpenAI-compatible server for the Planner, Prioritizer and scene knowledge, e.g.
    http://127.0.0.1:8000/v1 for a local vLLM server. Defaults to OpenRouter.
    (The coding agent and the Visual Differencing Module use --server-url.)"""

    planner_prompt: str | None = None
    """Planner system prompt. Defaults to prompts/planner.txt (prompts/joint.txt for physcap_joint)."""

    prioritizer_prompt: str | None = None
    """Prioritizer system prompt. Defaults to prompts/prioritizer.txt."""

    task_prompt: str | None = None
    """Task description given to the Planner and Prioritizer. Defaults to task.txt in the YAML's prompt_dir."""

    max_turns: int | None = None
    """Maximum turns per trial. Defaults to the YAML's max_turns, then 10."""

    run_label: str | None = None
    """Name of the results folder for this run. Defaults to the method name."""

    record_trial: bool = False
    """Film real-robot trials with an external USB webcam (see physcap/trial.py)."""

    key_file: str = ".openrouterkey"
    """File holding the OpenRouter API key, used when --api-key and OPENROUTER_API_KEY are unset."""

    start_proxy: bool = True
    """Start the local OpenRouter proxy on the --server-url port if nothing answers there."""


_DEFAULT_MODEL = next(f.default for f in dataclasses.fields(PhysCaPArgs) if f.name == "model")


def build(args: PhysCaPArgs) -> tuple[dict[str, Any], dict[str, Any], list | None]:
    """Resolve the task YAML and the method into CaP-X's (env_factory, config, api_servers)."""
    yaml_cfg = DictLoader.load([os.path.expanduser(args.config_path)])
    env_factory, config, api_servers = _load_config(args)

    # The task's prompts come from its folder in prompts/tasks/ (see physcap/prompt_files.py).
    prompts = (
        task_prompts(yaml_cfg["prompt_dir"], yaml_cfg.get("robot_prompt")) if yaml_cfg.get("prompt_dir") else {}
    )
    for key in ("prompt", "task_only_prompt", "multi_turn_prompt"):
        if key in prompts:
            env_factory["cfg"][key] = prompts[key]

    mode, with_physx = METHODS[args.method]
    if not with_physx:
        cfg = env_factory["cfg"]
        cfg["apis"] = [WITHOUT_PHYSX.get(name, name) for name in cfg["apis"]]

    # The config's `model:` fills in when --model is left at its default (CaP-X
    # merges the server URL the same way but does not read `model:` itself).
    if yaml_cfg.get("model") and args.model == _DEFAULT_MODEL:
        args.model = yaml_cfg["model"]
    args.planner_model = args.planner_model or yaml_cfg.get("planner_model") or args.model
    args.prioritizer_model = args.prioritizer_model or yaml_cfg.get("prioritizer_model") or args.model
    task_name = yaml_cfg.get("task_name") or Path(args.config_path).parent.name
    run_label = args.run_label or args.method

    config.update({
        "task_name": task_name,
        "method": args.method,
        "reasoning_mode": mode,
        "run_label": run_label,
        "planner_prompt": args.planner_prompt
            or str(PROMPTS_DIR / ("joint.txt" if mode == "joint" else "planner.txt")),
        "prioritizer_prompt": args.prioritizer_prompt or str(PROMPTS_DIR / "prioritizer.txt"),
        "task_prompt": args.task_prompt or yaml_cfg.get("task_prompt") or prompts.get("task_prompt_path"),
        "robot": prompts.get("robot"),
        "max_turns": args.max_turns or yaml_cfg.get("max_turns"),
        "record_trial": args.record_trial,
    })

    # outputs/<task>/<method>/<model>/<timestamp>/trial_NN  (CaP-X inserts the model name)
    base = config.get("output_dir") or f"outputs/{task_name}"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["output_dir"] = f"{base}/{run_label}/{stamp}"
    return env_factory, config, api_servers


def main(args: PhysCaPArgs) -> None:
    import capx.envs.runner as runner
    from physcap.summarize import print_run_summary
    from physcap.trial import run_trial

    args.api_key = _resolve_api_key(args)
    if args.agent_server_url:
        os.environ[AGENT_SERVER_URL_ENV] = args.agent_server_url.removesuffix("/chat/completions")
    # The dummy/real envs resolve free-form object names through this proxy too.
    os.environ.setdefault("PHYSCAP_SERVER_URL", args.server_url)
    env_factory, config, api_servers = build(args)
    if config.get("web_ui"):
        raise SystemExit("The web UI does not support PhysCaP's Planner/Prioritizer; run headless.")

    proxy = _ensure_proxy(args) if args.start_proxy else None
    runner._run_single_trial = run_trial
    server_procs = runner._start_api_servers(api_servers)
    try:
        runner._run_headless_trials(args, env_factory, config, time.time())
    finally:
        runner._stop_api_servers(server_procs)
        if proxy is not None:
            proxy.terminate()
    print_run_summary(config["output_dir"])


def _resolve_api_key(args: PhysCaPArgs) -> str | None:
    if args.api_key:
        return args.api_key
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    path = Path(args.key_file)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return None


def _ensure_proxy(args: PhysCaPArgs) -> subprocess.Popen | None:
    """Start CaP-X's OpenRouter proxy (``capx.serving.openrouter_server``) if the --server-url port is free."""
    base = args.server_url.rsplit("/chat/completions", 1)[0]
    if not base.startswith(("http://127.0.0.1", "http://localhost")):
        return None
    try:
        urllib.request.urlopen(base + "/health", timeout=2)
        return None
    except Exception:
        pass
    port = base.rsplit(":", 1)[-1]
    cmd = [sys.executable, "-m", "capx.serving.openrouter_server", "--key-file", args.key_file, "--port", port]
    print(f"[physcap] starting the OpenRouter proxy on port {port}")
    proc = subprocess.Popen(cmd)
    for _ in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen(base + "/health", timeout=2)
            return proc
        except Exception:
            continue
    proc.terminate()
    raise SystemExit(f"The OpenRouter proxy did not start on port {port}.")


if __name__ == "__main__":
    main(tyro.cli(PhysCaPArgs))
