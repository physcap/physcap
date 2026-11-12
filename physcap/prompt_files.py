"""Where the prompts live, and how a task's prompts are put together.

Every prompt the agents use is a plain-text file under ``prompts/`` at the
repository root (see ``prompts/README.md``):

    prompts/planner.txt, prioritizer.txt, joint.txt, scene_knowledge.txt
    prompts/coding_agent/*.txt            what the coding agent is told each turn
    prompts/tasks/<task>/task.txt         the task prompt (paper Appendix A.2)
    prompts/tasks/<task>/coding_rules.txt task-specific rules for the coding agent
    prompts/tasks/<task>/short.txt        one-line task summary
    prompts/tasks/<task>/robot.txt        (optional) which robot the agents control;
                                          default prompts/coding_agent/robot.txt
    prompts/robots/*.txt                  other robots, for tasks that reuse another
                                          task's prompts on a different robot

A task config names its folder with ``prompt_dir: prompts/tasks/<task>``, and can
name a robot file with ``robot_prompt:`` (for example the LIBERO simulation, which
uses the real Identify Empty Can prompts with ``prompts/robots/libero_franka.txt``).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
CODING_AGENT_DIR = PROMPTS_DIR / "coding_agent"

# The robot the paper's prompts name. A task with its own robot.txt gets that
# description instead, in the coding agent's prompt and in the Planner,
# Prioritizer and PhysCaP-joint prompts (see for_robot).
DEFAULT_ROBOT = "an AgileX PiPER 6-DOF robot arm"


def read_prompt(path: str | Path) -> str:
    """Read a prompt file. Relative paths are taken from the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.read_text(encoding="utf-8")


def coding_agent_prompt(name: str) -> str:
    """Read ``prompts/coding_agent/<name>.txt``."""
    return read_prompt(CODING_AGENT_DIR / f"{name}.txt")


def for_robot(text: str, robot: str | None) -> str:
    """Name ``robot`` instead of the AgileX PiPER in a Planner or Prioritizer prompt.

    The prompt files keep the paper's wording; for a task on another robot (for
    example the LIBERO simulation) the robot's description is swapped in when
    the prompt is loaded, including the robot named in the Prioritizer's table
    of action times.
    """
    if not robot or robot == DEFAULT_ROBOT:
        return text
    return text.replace(DEFAULT_ROBOT, robot).replace("(AgileX PiPER)", f"({robot})")


def task_prompts(prompt_dir: str | Path, robot_prompt: str | Path | None = None) -> dict[str, str]:
    """Build everything a task needs from its prompt folder.

    ``robot_prompt`` is a file describing the robot; when given, it is used
    instead of the folder's ``robot.txt``.

    Returns:
        task_prompt_path: path of ``task.txt``, read by the Planner and Prioritizer.
        prompt: the coding agent's first message, ``coding_agent/task_template.txt``
            filled with the robot line, ``task.txt`` and ``coding_rules.txt``.
        robot: which robot the agents control (``robot_prompt``, else ``robot.txt``).
        task_only_prompt: ``short.txt``, used by the Visual Differencing Module.
        multi_turn_prompt: ``coding_agent/multi_turn.txt``, CaP-X's follow-up message
            for the CaP and CaP+PhysX methods.
    """
    prompt_dir = Path(prompt_dir)
    if not prompt_dir.is_absolute():
        prompt_dir = REPO_ROOT / prompt_dir
    task = read_prompt(prompt_dir / "task.txt").strip()
    rules = read_prompt(prompt_dir / "coding_rules.txt").strip()
    robot_file = prompt_dir / "robot.txt"
    if robot_prompt is None:
        robot_prompt = robot_file if robot_file.exists() else CODING_AGENT_DIR / "robot.txt"
    robot = read_prompt(robot_prompt).strip()
    return {
        "task_prompt_path": str(prompt_dir / "task.txt"),
        "robot": robot,
        "prompt": coding_agent_prompt("task_template").format(robot=robot, task=task, coding_rules=rules),
        "task_only_prompt": read_prompt(prompt_dir / "short.txt").strip(),
        "multi_turn_prompt": coding_agent_prompt("multi_turn"),
    }
