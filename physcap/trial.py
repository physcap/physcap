"""One PhysCaP trial.

``run_trial`` has the same signature as CaP-X's ``capx.envs.trial._run_single_trial``
and replaces it in the CaP-X runner (see ``physcap.run``). It supports every
method in the paper:

- ``none`` (CaP, CaP+PhysX): the coding agent works alone. This is CaP-X's own
  multi-turn loop, called unchanged.
- ``planner`` (CaP+PhysX+Planner), ``joint`` (PhysCaP-joint) and
  ``planner_prioritizer`` (PhysCaP): each turn the Planner decides whether more
  information is needed and lists exploration candidates, the Prioritizer (if
  used) ranks them, and the coding agent writes code for the chosen candidate
  only. Measurements are written to the scene knowledge, which the next turn
  reads.

Every trial gets its own folder, ``<output_dir>/trial_NN/``, holding the
Planner/Prioritizer outputs (``reasoning/``), ``measurement_report.json``,
``time_log.txt`` and, inside it, the usual CaP-X artifact folder.
"""

from __future__ import annotations

import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

import capx.envs.trial as capx_trial
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import (
    _annotate_code_blocks,
    _build_log_lines,
    _capture_initial_visual_feedback,
    _get_visual_differencing_feedback,
    _patch_libero_goal,
    _save_trial_video,
    _save_turn_and_combined_videos,
)
from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model,
    query_model_ensemble,
    query_single_model_ensemble,
)
from capx.utils.launch_utils import TrialSummary, _extract_code, _save_trial_artifacts

from physcap.timing import TrialTimer, write_time_log

REASONING_MODES = ("none", "planner", "joint", "planner_prioritizer")

# OpenCV index of the external webcam that films real-robot trials. It depends
# on how the machine is wired, not on the experiment, so it is not a CLI flag.
# List the cameras with `v4l2-ctl --list-devices` (Linux) to find it.
RECORD_TRIAL_CAMERA_INDEX = 2

DEFAULT_MAX_TURNS = 10


def run_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Run one trial with the method named by ``config["reasoning_mode"]``."""
    mode = config.get("reasoning_mode", "none")
    if mode not in REASONING_MODES:
        raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {mode!r}")

    trial_dir = Path(config["output_dir"]) / f"trial_{trial:02d}" if config.get("output_dir") else None
    trial_config = dict(config)
    if trial_dir is not None:
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_config["output_dir"] = str(trial_dir)

    start = time.time()
    timer = TrialTimer()
    webcam = _start_webcam(config, trial_dir)
    max_turns = int(config.get("max_turns") or DEFAULT_MAX_TURNS)

    try:
        if mode == "none":
            capx_trial.MULTITURN_LIMIT = max_turns
            summary = capx_trial._run_single_trial(
                env, trial, args, trial_config, multi_turn_prompt, partial_artifacts=partial_artifacts
            )
        else:
            summary = _run_reasoning_trial(
                env, trial, args, trial_config, mode, max_turns, timer, partial_artifacts
            )
    finally:
        webcam_info = _stop_webcam(webcam, start)

    _collect_step_timings(env, timer)
    trial_total = time.time() - start
    print(f"Trial {trial} took {trial_total:.2f} seconds")
    if trial_dir is not None:
        times = write_time_log(timer, trial_dir / "time_log.txt", trial_total, webcam_info)
        write_measurement_report(env, trial, args, config, trial_dir, times)
    gc.collect()
    return summary


# ---------------------------------------------------------------------------
# Planner / Prioritizer loop
# ---------------------------------------------------------------------------

def _run_reasoning_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    mode: str,
    max_turns: int,
    timer: TrialTimer,
    partial_artifacts: dict[str, Any] | None,
) -> TrialSummary:
    from physcap.agents.codegen_prompt import build_candidate_code_prompt
    from physcap.agents.cycle import run_reasoning_turn
    from physcap.agents.scene_knowledge import (
        SCENE_KNOWLEDGE_FILENAME,
        append_stdout_to_knowledge,
        init_scene_knowledge,
        load_object_labels,
        save_scene_image,
        update_scene_knowledge,
    )

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    t0 = time.time()
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    timer.record("A1_env_reset", time.time() - t0)
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)
    if (config["record_video"] or use_video_diff) and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    recording_frames = (config["record_video"] or use_video_diff) and hasattr(env, "get_video_frame_count")

    vdm_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )
    if config["use_img_differencing"] or use_video_diff:
        assert vdm_args.model in VLM_MODELS, "The visual differencing model must be listed in VLM_MODELS"

    t0 = time.time()
    visual_feedback_imgs, image_history, task_description = _capture_initial_visual_feedback(
        env, obs, config, args, vdm_args
    )
    if not task_description:
        task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])
    wrist_history: list[str] | None = [] if use_wrist else None
    if wrist_history is not None:
        _append_if(wrist_history, _render_wrist_b64(env))
    timer.record("A2_initial_scene", time.time() - t0)

    knowledge_dir = Path(config["output_dir"]) / "reasoning" if config.get("output_dir") else None
    if knowledge_dir is not None:
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        # The measurement APIs write each reading into the scene knowledge through this attribute.
        env.knowledge_dir = str(knowledge_dir)

    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    turn_frame_ranges: list[tuple[int, int]] = []
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False}
    reward, terminated, truncated = 0.0, False, False
    num_regenerations = num_finishes = 0
    raw_code = ""

    if partial_artifacts is not None:
        partial_artifacts.update({
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": 0,
            "num_finishes": 0,
            "num_code_blocks": 0,
            "ensemble_data": None,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Describe the scene once before the first turn; every agent reads this file.
    if knowledge_dir is not None:
        _, first_frame = _render(env)
        if first_frame is not None:
            scene_path = save_scene_image(first_frame, knowledge_dir / "turn_00", "scene.png")
            t0 = time.time()
            try:
                init_scene_knowledge(
                    image_path=Path(scene_path),
                    task=task_description,
                    output_dir=knowledge_dir,
                    api_key=args.api_key,
                    model_name=args.planner_model,
                )
            except Exception as exc:
                print(f"[physcap] Warning: could not initialise scene knowledge: {exc}")
            timer.record("B1_scene_knowledge_init", time.time() - t0)

    turn = 0
    while turn < max_turns:
        turn += 1
        print(f"[physcap] -- turn {turn} --")
        frame_b64, frame = _render(env)
        if frame is None:
            raise RuntimeError("The environment returned no camera frame; the Planner needs one.")
        visual_feedback_imgs.append(frame)
        image_history.append(frame_b64)
        if wrist_history is not None:
            _append_if(wrist_history, _render_wrist_b64(env))

        turn_dir = (knowledge_dir or Path(config.get("output_dir") or ".") / "reasoning") / f"turn_{turn:02d}"
        scene_path = save_scene_image(frame, turn_dir, "scene.png")

        previous_result = None
        if turn > 1 and all_responses:
            previous_result = {
                "executed_plan_name": all_responses[-1].get("selected_candidate", {}).get("name"),
                "executed_candidate": all_responses[-1].get("selected_candidate"),
                "success": info_step.get("sandbox_rc", -1) == 0,
                "sandbox_rc": info_step.get("sandbox_rc", -1),
                "stdout": info_step.get("stdout", ""),
                "stderr": info_step.get("stderr", ""),
            }

        reasoning = run_reasoning_turn(
            image_path=scene_path,
            task=task_description,
            output_dir=knowledge_dir or turn_dir,
            turn_dir=turn_dir,
            iteration=turn - 1,
            execution_result=previous_result,
            api_key=args.api_key,
            planner_model=args.planner_model,
            prioritizer_model=args.prioritizer_model,
            use_prioritizer=mode == "planner_prioritizer",
            follow_fresh_ordering=mode == "joint",
            planner_system_prompt_path=config.get("planner_prompt"),
            prioritizer_system_prompt_path=config.get("prioritizer_prompt"),
            user_task_prompt_path=config.get("task_prompt"),
            robot=config.get("robot"),
            debug=args.debug,
        )
        timer.record("B2_planner", reasoning.get("timing", {}).get("planner_s", 0.0), turn=turn)
        timer.record("B3_prioritizer", reasoning.get("timing", {}).get("prioritizer_s", 0.0), turn=turn)

        candidate = reasoning["selected_candidate"]
        if not candidate:
            raise RuntimeError("The Planner returned no candidate.")
        name = candidate.get("name", "")
        response = {
            "turn": turn,
            "selected_candidate": candidate,
            "reasoning_candidates_path": reasoning["candidates_json_path"],
            "reasoning_prioritized_path": reasoning["prioritized_json_path"],
        }

        if name == "task_complete":
            # The Planner sees the goal already reached; writing code now would redo it.
            all_responses.append({**response, "decision": "task_complete"})
            num_finishes += 1
            print(f"[physcap] turn {turn}: the Planner reports the task is complete.")
            break

        final_step = name == "original_task_execute"
        print(f"[physcap] turn {turn}: " + ("enough information, generating code for the whole task"
                                            if final_step else f"exploring: {name}"))

        differencing = ""
        if config.get("use_img_differencing") and len(image_history) >= 2:
            differencing = _get_visual_differencing_feedback(
                vdm_args, task_description, image_history, wrist_base64_history=wrist_history
            ) or ""

        knowledge_text = None
        if final_step and knowledge_dir is not None and (knowledge_dir / SCENE_KNOWLEDGE_FILENAME).exists():
            knowledge_text = (knowledge_dir / SCENE_KNOWLEDGE_FILENAME).read_text(encoding="utf-8")

        code_prompt = build_candidate_code_prompt(
            obs,
            candidate,
            original_task=task_description,
            executed_code="\n\n".join(code_blocks),
            console_stdout=info_step.get("stdout", ""),
            console_stderr=info_step.get("stderr", ""),
            differencing_feedback=differencing,
            scene_knowledge=knowledge_text,
            object_labels=load_object_labels(knowledge_dir) if knowledge_dir is not None else [],
        )

        t0 = time.time()
        raw_code, reasoning_text, turn_ensemble = _query_code(args, config, code_prompt)
        timer.record("B4_codegen", time.time() - t0, turn=turn)
        if turn == 1:
            ensemble_data = turn_ensemble
        elif turn_ensemble is not None:
            multiturn_ensemble_data.append({**turn_ensemble, "regeneration": turn - 1})

        blocks = _extract_code(raw_code or "")
        code = blocks[0] if blocks else (raw_code or "").strip()
        code_blocks.append(code)
        code_block_metadata.append({"generation": turn - 1, "regenerated": turn > 1})
        all_responses.append({
            **response,
            "decision": "candidate_codegen",
            "reasoning_prompt_task": task_description,
            "code_generation_prompt": code_prompt if config.get("save_multiturn_prompts") else None,
            "raw_code_response": raw_code,
            "reasoning": reasoning_text or "",
            "code_blocks": blocks,
            "interaction_history": reasoning.get("interaction_history", {}),
        })
        if partial_artifacts is not None:
            partial_artifacts["raw_code"] = raw_code

        frame_start = env.get_video_frame_count() if recording_frames else 0
        t0 = time.time()
        obs, reward, terminated, truncated, info_step = env.step(code)
        if not hasattr(env, "step_timings"):
            timer.record("C_total", time.time() - t0, turn=turn)
        _collect_step_timings(env, timer)
        turn_frame_ranges.append((frame_start, env.get_video_frame_count() if recording_frames else 0))

        if knowledge_dir is not None:
            # Printed values first, then what changed in the image.
            append_stdout_to_knowledge(
                output_dir=knowledge_dir, stdout=info_step.get("stdout", ""), turn=turn, candidate_name=name
            )
            _, after = _render(env)
            if after is not None:
                t0 = time.time()
                try:
                    update_scene_knowledge(
                        output_dir=knowledge_dir,
                        previous_image=frame,
                        current_image=after,
                        action_description=_describe_action(candidate),
                        visual_differencing_args=vdm_args,
                        task_description=task_description,
                    )
                except Exception as exc:
                    print(f"[physcap] Warning: could not update scene knowledge: {exc}")
                timer.record("B5_knowledge_update", time.time() - t0, turn=turn)

        if info_step.get("stderr"):
            stderr_history.append(info_step["stderr"])
        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "num_regenerations": turn - 1,
                "num_code_blocks": len(code_blocks),
            })

        finished = (
            final_step
            or info_step.get("task_completed", False)
            or terminated
            or truncated
            or "FINISH" in info_step.get("stdout", "")
        )
        if finished:
            num_finishes += 1
            break
        num_regenerations += 1

    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    stderr = "\n\n".join(stderr_history) if stderr_history else info_step.get("stderr", "")
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, len(code_blocks), stderr_override=stderr,
    )

    t0 = time.time()
    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward, info_step.get("task_completed", False),
        final_code, code_blocks[-1] if code_blocks else None, all_responses, log_lines,
        visual_feedback_imgs, ensemble_data=ensemble_data, multiturn_ensemble_data=multiturn_ensemble_data,
    )
    timer.record("D1_artifact_save", time.time() - t0)
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(env, config, trial, info_step, reward, turn_frame_ranges)
    else:
        _save_trial_video(env, config, trial, info_step, reward, len(code_blocks))

    return TrialSummary(
        trial=trial,
        success=info_step["sandbox_rc"] == 0,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=len(code_blocks),
    )


# ---------------------------------------------------------------------------
# Measurement report
# ---------------------------------------------------------------------------

def write_measurement_report(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    trial_dir: Path,
    times: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Write ``measurement_report.json``: which objects were measured, in what order.

    The number of measurements is the paper's OI (objects interacted). Only the
    PiPER envs (real and dummy) keep this record; other envs are skipped.
    """
    low_level = getattr(env, "low_level_env", None)
    build = getattr(low_level, "measurement_report", None)
    if not callable(build):
        return None
    try:
        body = build()
    except Exception as exc:
        print(f"[physcap] Warning: could not build the measurement report: {exc}")
        return None

    settings = {}
    for key, reader in (("mass_config", "mass_estimator_config"), ("stiffness_config", "physics_config")):
        read = getattr(low_level, reader, None)
        if callable(read):
            try:
                settings[key] = read()
            except Exception as exc:
                print(f"[physcap] Warning: could not read {reader}: {exc}")

    report = {
        "schema_version": body.get("schema_version"),
        "trial": trial,
        "task": config.get("task_name"),
        "method": config.get("method"),
        "reasoning_mode": config.get("reasoning_mode", "none"),
        "run_label": config.get("run_label"),
        "planner_prompt": Path(config["planner_prompt"]).name if config.get("planner_prompt") else None,
        "prioritizer_prompt": Path(config["prioritizer_prompt"]).name if config.get("prioritizer_prompt") else None,
        "model": getattr(args, "model", None),
        "planner_model": getattr(args, "planner_model", None),
        "prioritizer_model": getattr(args, "prioritizer_model", None),
        "times": times,
        **settings,
        **{k: v for k, v in body.items() if k != "schema_version"},
    }
    path = trial_dir / "measurement_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    totals = report.get("totals", {})
    print(f"[physcap] {totals.get('measurements')} measurement(s); order={report.get('order')} -> {path}")
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_code(args, config: dict[str, Any], prompt: list[dict]) -> tuple[str, str | None, dict | None]:
    if not config.get("use_parallel_ensemble"):
        out = query_model(args, prompt)
        return out["content"], out.get("reasoning"), None
    if config.get("use_multimodel"):
        out = query_model_ensemble(args, prompt, is_multiturn=False)
    else:
        out = query_single_model_ensemble(args, prompt, args.model, is_multiturn=False)
    ensemble = {
        "ensemble_candidates_txt": out.get("ensemble_candidates_txt", ""),
        "ensemble_synthesis_txt": out.get("ensemble_synthesis_txt", ""),
    }
    return out["content"], out.get("reasoning"), ensemble


def _render(env: CodeExecutionEnvBase):
    """Return ``(base64 data URL, PIL image)`` of the current camera frame, or ``(None, None)``."""
    import base64
    import io

    from PIL import Image

    img = env.render() if hasattr(env, "render") else None
    if img is None:
        return None, None
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"), pil


def _render_wrist_b64(env: CodeExecutionEnvBase) -> str | None:
    import base64
    import io

    from PIL import Image

    img = env.render_wrist() if hasattr(env, "render_wrist") else None
    if img is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _append_if(history: list[str], item: str | None) -> None:
    if item is not None:
        history.append(item)


def _describe_action(candidate: dict[str, Any]) -> str:
    name = candidate.get("name", "")
    desc = (candidate.get("description", "") or "").split("\n")[0][:300]
    if name and desc and name != desc:
        return f"{name}: {desc}"
    return desc or name or "Unknown action"


def _collect_step_timings(env: CodeExecutionEnvBase, timer: TrialTimer) -> None:
    """Move the per-step timings ``PiperEnv`` records into the trial timer.

    ``PiperEnv.step`` stores its own wall time and the robot sub-timings
    (perception, motion, gripper, measurement) per turn, so this works the same
    whether the steps ran in the Planner loop or inside CaP-X's own loop.
    """
    for turn, seconds in getattr(env, "step_timings", None) or []:
        timer.record("C_total", seconds, turn=turn)
    for turn, category, seconds in getattr(env, "step_events", None) or []:
        timer.record(category, seconds, turn=turn)
    for name in ("step_timings", "step_events"):
        if hasattr(env, name):
            getattr(env, name).clear()


def _start_webcam(config: dict[str, Any], trial_dir: Path | None):
    if not config.get("record_trial") or trial_dir is None:
        return None
    try:
        from physcap.hardware.webcam_recorder import WebcamRecorder
    except ImportError:
        print("[physcap] Warning: --record-trial needs physcap.hardware, which is not installed.")
        return None

    recorder = WebcamRecorder(output_path=trial_dir / "trial_recording.mp4", camera_index=RECORD_TRIAL_CAMERA_INDEX)
    try:
        recorder.start()
    except RuntimeError as exc:
        print(f"[physcap] Warning: webcam not recording: {exc}")
        return None
    return recorder


def _stop_webcam(recorder, trial_start: float) -> dict | None:
    if recorder is None:
        return None
    recorder.stop()
    recorder.release()
    return recorder.summary(trial_start_wall=trial_start)
