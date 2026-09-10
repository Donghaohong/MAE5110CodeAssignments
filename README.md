# MAE 5110 Code Assignments

Code assignments for MAE 5110.

## Installation

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). After cloning this repository, run the following command from its root directory:

```console
uv sync --python 3.14
```

This creates a local `.venv` and installs the required dependencies. Run Python commands inside the environment with `uv run`, for example:

```console
uv run python assignment_0.py
```

## Assignments

- [Assignment 0](assignments/assignment_0.md)
- [Assignment 1](assignments/assignment_1.md)

## HW1: Rimless Wheel

From the repository root on the HW1 branch, run the commands below in order.
Git and `uv` must be installed as described above.

HW1 scripts, reports, and results are in `HW1/`. The shared `models/` directory
and `integrators.py` stay at the repository root. Use `-m HW1.<module>` from
the root so both the shared code and HW1 modules can be imported correctly.

Run the original single-state, fixed-step simulation:

```bash
uv run python -m HW1.assignment_1
```

Compute the 61 x 61 RoA grid:

```bash
uv run python -m HW1.rimless_wheel_roa
```

Save and display the RoA plot (close the figure window to finish):

```bash
uv run python -m HW1.plot_rimless_wheel_roa
```

Generate and display the return map (close the figure window to finish):

```bash
uv run python -m HW1.rimless_wheel_return_map
```

Estimate the Floquet multiplier using the saved return-map result:

```bash
uv run python -m HW1.rimless_wheel_floquet
```

Sweep slopes and spoke counts using a 31 x 31 grid per case:

```bash
uv run python -m HW1.rimless_wheel_parameter_sweep
```

The scripts save results beside the code in `HW1/`:

- **RoA:** `rimless_wheel_roa_event.png` and `rimless_wheel_roa_results_event.npz`.
- **Return map:** `rimless_wheel_return_map.png` and `rimless_wheel_return_map_results.npz`. The Floquet script reads this data and prints a multiplier of approximately **0.5** for the default parameters.
- **Parameter sweeps:** `rimless_wheel_slope_sweep_roa.png`, `rimless_wheel_spokes_sweep_roa.png`, `rimless_wheel_sweep_summary.png`, and `rimless_wheel_parameter_sweep_results.npz`.

See [the HW1 report](HW1/assignment_1_report.md) for the sanity checks, figures, and discussion.
