# Install

## Requirements

- **Linux on an x86-64 machine.** CaP-X, which PhysCaP is built on, only installs there.
- **[git](https://git-scm.com/)** and **[uv](https://docs.astral.sh/uv/)**, the Python package manager used here. uv installs the right Python version (3.10) by itself.
- **Disk space:** about 6 GB for the code (most of it is CaP-X and the libraries it includes as submodules) and about 7.5 GB for the Python environment.
- **An [OpenRouter](https://openrouter.ai/keys) API key.** All agents call their models through OpenRouter, and every call costs money.

Dummy mode does not need a GPU or a robot: the models are called over the internet, and the robot is replaced by a photo and a table of values. The optional [LIBERO simulation](libero.md) needs a GPU that can render without a display (EGL).

## 1. Get the code

```bash
git clone --recurse-submodules https://github.com/physcap/physcap.git
cd physcap
```

`--recurse-submodules` also downloads CaP-X into `third_party/cap-x`, with the libraries CaP-X itself includes. If you cloned without it, run:

```bash
git submodule update --init --recursive
```

## 2. Install

If uv is not installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, in the repository folder:

```bash
uv sync
```

This creates `.venv/` with Python 3.10, PhysCaP and CaP-X, at the exact package versions in `uv.lock`. The first run downloads a few GB and can take a while.

For the LIBERO simulation, add `--extra libero` ([LIBERO simulation](libero.md)). To also run the tests, add the development tools:

```bash
uv sync --extra dev
```

## 3. Add your API key

Put your OpenRouter key in a file named `.openrouterkey` in the repository folder. Git ignores this file, so it is never committed.

```bash
echo "sk-or-v1-your-key-here" > .openrouterkey
```

Instead of the file you can set the `OPENROUTER_API_KEY` environment variable, or pass `--api-key` on the command line.

## 4. Check the install

The tests run offline. They make no model calls and cost nothing:

```bash
uv run --extra dev python -m pytest tests -q
```

If LIBERO is installed, this includes a scripted LIBERO check that takes a few minutes; without LIBERO it is skipped.

Then run one trial in dummy mode (this makes paid model calls, a few minutes per trial):

```bash
uv run python -m physcap.run --config-path configs/identify_empty_can/dummy.yaml --method physcap
```

`physcap.run` first starts a small local server (the CaP-X OpenRouter proxy, on port 8110) that forwards the coding agent's model calls to OpenRouter, and stops it at the end. When the trial ends it prints a table like this:

```
task                  method                 n     SR   OI              Time (s)
------------------------------------------------------------------------------------------
identify_empty_can    physcap                1   100%   2.00            184.03
```

SR is the success rate (a trial succeeds when the agent's declared answer is correct), OI (objects interacted) is the number of measurements the robot made, counting a repeated measurement of the same object again, and Time is the trial length. With more than one trial, OI and Time also show the standard deviation. The trial's files are under `outputs/identify_empty_can/physcap/`. [Running experiments](reproducing_results.md) explains them.

## Updating

```bash
git pull
git submodule update --init --recursive
uv sync
```

## Troubleshooting

- **Messages such as `Robosuite not installed!`, `LIBERO not installed!` or a `No module named 'robosuite'` traceback when a run starts.** These are normal: CaP-X reports the simulators it can use, and dummy mode needs none of them.

- **`uv sync` says a folder in `third_party/` "does not appear to be a Python project".** The submodules are missing. Run `git submodule update --init --recursive`.
- **`No module named pytest`.** Install the development tools with `uv sync --extra dev`.
- **The trial stops with an OpenRouter error about the model.** Check that your key has credit and that the model name in the config (`model:`) is still offered by OpenRouter. See [Customizing](customizing.md#switch-models) to use another model.
