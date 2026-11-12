# PhysCaP

### Grounding Code-as-Policy Agent with Physics-Informed Exploration

[Paper (arXiv)](https://arxiv.org/abs/2608.21031) &ensp;|&ensp; [Project page](https://physcap.github.io)

Accepted at the [8th Robot Learning Workshop @ NeurIPS 2026](http://www.robot-learning.ml/2026/) and the [Workshop on Agentic Robotics @ CoRL 2026](https://agentic-robotics-workshop.github.io/).

PhysCaP is a Physics-Informed Code-as-Policy agent system for active perception in robot manipulation. In code-as-policy, a language model (the **coding agent**) controls the robot by writing Python code that calls robot tools. PhysCaP adds tools that measure an object's mass and stiffness with the robot itself (the PhysX modules), and two agents that decide how to use them: a **Planner** that decides when to explore and when to stop, and a **Prioritizer** that filters and ranks the possible interactions so the robot explores with as few interactions as possible.

This repository contains the Planner, the Prioritizer and the coding agent, and a **dummy mode** that runs them without a robot: a photo stands in for the camera and measurements come from a table of true values. The real-robot code, including the PhysX measurement modules, is not included yet, so the paper's real-robot numbers (Table 1) cannot be reproduced here; dummy mode reproduces how the agents decide. A [LIBERO simulation](docs/libero.md) of Identify Empty Can, with the real task's cans, tray and prompts, runs the whole method, including grasping and lifting each can to measure it. It is built on [CaP-X](https://github.com/capgym/cap-x), which is included as a git submodule in `third_party/cap-x`.

## Quick start

```bash
git clone --recurse-submodules https://github.com/physcap/physcap.git && cd physcap
uv sync
echo "sk-or-v1-your-key-here" > .openrouterkey          # an OpenRouter API key
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap
```

The last command makes paid model calls and takes a few minutes. The full steps are in [docs/install.md](docs/install.md).

## Documentation

| Guide | What it covers |
| --- | --- |
| [Install](docs/install.md) | Requirements, installing, the API key, checking the install. |
| [Method](docs/method.md) | What happens in one trial: the Planner, the Prioritizer and the coding agent, with the paper section for each part. |
| [Dummy mode](docs/dummy_mode.md) | Running the tasks without a robot: photos, ground-truth values, grading. |
| [Running experiments](docs/reproducing_results.md) | Commands for every task and method, where results go, and how to summarise them. |
| [LIBERO simulation](docs/libero.md) | Identify Empty Can in simulation, with physical measurement. |
| [Customizing](docs/customizing.md) | Adding a task, adding a measurement, editing prompts, switching models. |
| [Prompts](prompts/README.md) | Every prompt the agents use, as plain-text files. |

## Tasks and methods

| Paper task | Config folder | In this repository |
| --- | --- | --- |
| Identify Empty Can | `configs/identify_empty_can` | dummy mode |
| Pack Grocery Bag | `configs/pack_grocery_bag` | dummy mode |
| Identify Empty Can, simulated | `configs/libero_identify_empty_can` | LIBERO simulation |

| Paper method | `--method` | What the coding agent has |
| --- | --- | --- |
| CaP | `cap` | The coding agent alone, with no measurement tools. |
| CaP+PhysX | `cap_physx` | The measurement tools `get_mass` and `get_stiffness`. |
| CaP+PhysX+Planner | `cap_physx_planner` | A Planner that decides what to explore and when to stop. |
| PhysCaP-joint | `physcap_joint` | The Planner and the Prioritizer merged into one agent. |
| PhysCaP | `physcap` | The Planner, then a separate Prioritizer. |

## Repository layout

| Path | Contents |
| --- | --- |
| `physcap/` | The PhysCaP code. `run.py` is the command line, `trial.py` runs one trial, `agents/` holds the Planner and Prioritizer, `apis/` the tools the coding agent can call, `envs/` the dummy environment that stands in for the robot (the paper's AgileX PiPER arm) and the LIBERO simulation. |
| `prompts/` | Every prompt, as plain text. |
| `configs/` | One folder per task: settings, ground-truth values, scene photos. |
| `docs/` | The guides above. |
| `tests/` | Offline tests (no robot, no model calls). |
| `third_party/cap-x/` | CaP-X, pinned to a fixed version ([physcap/cap-x](https://github.com/physcap/cap-x)). |

## Coming soon

- **PhysX for the AgileX PiPER**: the modules that measure an object's mass and stiffness with the real robot, and the settings for running PhysCaP on it.

## Citation

```bibtex
@article{lin2026physcap,
  title   = {PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration},
  author  = {Lin, Chen-Yu and Chen, Jing-Wen and Chang, Hsueh-En and Chen, Hung-An and
             Chang, Sheng-Hsun and Huang, Chi-Pin and Yang, Fu-En and Chen, Min-Hung and
             Chen, Yi-Ting and Wang, Yu-Chiang Frank and Sun, Shao-Hua},
  journal = {arXiv preprint arXiv:2608.21031},
  year    = {2026}
}
```

PhysCaP is built on CaP-X ([arXiv:2603.22435](https://arxiv.org/abs/2603.22435)); please cite it too if you use the framework.

## License

Released under the [MIT License](LICENSE). CaP-X is also under the MIT License; its notice is kept in `LICENSE` and in `third_party/cap-x`.
