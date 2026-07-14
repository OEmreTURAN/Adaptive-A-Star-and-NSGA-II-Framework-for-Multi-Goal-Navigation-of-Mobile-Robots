# Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots

This repository provides the implementation and publication figures associated with the article **"Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots."** The software evaluates A*-initialized, B-Spline-based multi-objective path planning for sequential mobile-robot navigation across five simulation environments (`Level-1` through `Level-5`).

## Project metadata

| Item | Details |
|---|---|
| Article title | Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots |
| Repository purpose | Reproduce the trajectory, comparative, sensitivity, timing, and ablation evaluations described in the article |
| Authors | Osman Emre Turan, Oguz Misir, and Mustafa Ozden |
| Corresponding author | Osman Emre Turan ([emre.turan@btu.edu.tr](mailto:emre.turan@btu.edu.tr)) |
| Reference Python version | Python 3.12.4 |

## Quick start

```bash
git clone https://github.com/OEmreTURAN/Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots.git
cd Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a one-episode smoke test of the proposed planner in `Level-1`:

```bash
python -u patrollingAlgorithms.py segment_timing 1 Level-1 proposed_only
```

A successful smoke test prints progress to the terminal and generates, in the current working directory, `calculation_times_segment_Level-1.json` together with timing-summary CSV, LaTeX-table, and Markdown-report files. This one-episode run verifies software execution only; it does not reproduce the article's statistical timing results.

## Repository structure

```text
.
|-- patrollingAlgorithms.py   # Implementation and experiment entry point
|-- requirements.txt          # Pinned Python dependencies
|-- README.md
`-- figures/
    |-- paths/                # Representative publication trajectories
    |-- boxplots/             # Publication box plots
    |-- raincloud/            # Publication raincloud plots
    |-- radar/                # Publication radar charts
    `-- sensitivity/          # Publication sensitivity figures
```


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OEmreTURAN/Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots.git
cd Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

When Python 3.12 is already the default interpreter, use `python -m venv .venv`.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned dependencies are:

- NumPy 1.26.4
- SciPy 1.14.0
- Matplotlib 3.9.0
- DEAP 1.4.3

A GPU is not required. Multi-episode evaluations use CPU multiprocessing when multiple logical processors are available.

### 4. Verify the environment

```bash
python -c "import numpy, scipy, matplotlib, deap; print('Dependencies imported successfully')"
python -m py_compile patrollingAlgorithms.py
```

## Command-line interface

Run commands from the repository root using:

```text
python -u patrollingAlgorithms.py <mode> [arguments]
```

The `-u` option prints progress without terminal-output buffering. All generated numerical files, reports, tables, and new figures are written to the current working directory.

Available modes:

| Mode | Purpose |
|---|---|
| `single` | Generate representative fixed-seed trajectories |
| `test` | Run the main 100-episode method comparison |
| `sensitivity` | Evaluate parameter sensitivity |
| `segment_timing` | Measure route and single-segment planning time |
| `segment_timing_aggregate` | Rebuild summaries from existing timing checkpoints |
| `ablation` | Run controlled module-ablation evaluations |

## Reproducing the article analyses

| Analysis | Full reproduction command | Generated output types |
|---|---|---|
| Representative trajectories | `python -u patrollingAlgorithms.py single` | PNG trajectory figures and timing JSON files |
| Main 100-episode comparison | `python -u patrollingAlgorithms.py test` | Per-environment JSON summaries and PNG box, raincloud, radar, normalized-bar, and metric plots |
| Parameter sensitivity | `python -u patrollingAlgorithms.py sensitivity 30 rho_tau` | JSON results, CSV summaries, LaTeX-table file, Markdown report, and PNG sensitivity figures |
| Route and single-segment timing | `python -u patrollingAlgorithms.py segment_timing 100 all proposed_only` | Per-level checkpoint JSON files, CSV summaries, LaTeX-table file, and Markdown timing report |
| Ablation analysis | `python -u patrollingAlgorithms.py ablation 30 all` | JSON results, CSV summary, LaTeX-table file, and Markdown ablation report |

The full reproduction commands are computationally intensive. Run them only when sufficient CPU time is available and retain their generated files in a dedicated working directory.

## 1. Representative trajectories

```bash
python -u patrollingAlgorithms.py single
```

This mode uses seed 50 and evaluates the proposed framework and comparison methods once across `Level-1` through `Level-5`. It generates representative trajectory figures and timing JSON files.

## 2. Main 100-episode comparison

```bash
python -u patrollingAlgorithms.py test
```

This mode runs 100 episodes per environment for all evaluated methods and records:

- path length;
- maximum curvature;
- minimum obstacle clearance;
- smoothness;
- wheel effort;
- centering; and
- computation time.

It generates per-environment JSON summaries and derived visual outputs, including box plots, raincloud plots, radar charts, normalized bar charts, and metric-specific plots.

## 3. Parameter sensitivity

Syntax:

```text
python -u patrollingAlgorithms.py sensitivity [episodes] [scope]
```

Available scopes:

| Scope | Parameters evaluated |
|---|---|
| `rho_tau` | A* seed ratio and softmin temperature |
| `rho` | A* seed ratio only |
| `tau` | Softmin temperature only |
| `all` | All implemented parameter sweeps |

Reproduce the article's rho/tau analysis:

```bash
python -u patrollingAlgorithms.py sensitivity 30 rho_tau
```

Run all implemented sensitivity sweeps:

```bash
python -u patrollingAlgorithms.py sensitivity 30 all
```

Sensitivity runs generate JSON results, CSV summaries, LaTeX-table data, a Markdown report, and sensitivity figures. These evaluations compare fixed parameter values; they do not implement online parameter adaptation.

## 4. Route and single-segment timing

Syntax:

```text
python -u patrollingAlgorithms.py segment_timing [episodes] [environment] [method-filter]
```

The environment argument can be `all` or one of `Level-1`, `Level-2`, `Level-3`, `Level-4`, or `Level-5`.

Method filters:

| Filter | Methods evaluated |
|---|---|
| `proposed_only` | Proposed NSGA-II framework |
| `competitors_only` | Standard GA, AB-WOA-APF, and HWPSO |
| `all` | Proposed framework and all comparison methods |

Reproduce the article's proposed-method timing analysis:

```bash
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
```

Run one level only:

```bash
python -u patrollingAlgorithms.py segment_timing 100 Level-3 proposed_only
```

The timing mode records complete-route runtime and individual waypoint-to-waypoint segment times. One proposed-method segment optimization jointly generates the Length, Smooth, Effort, Centered, Safe, and Adaptive route variants.

### Resume an interrupted timing run

Checkpoint files are updated during execution. To resume, run the same command from the same working directory:

```bash
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
```

Completed records are detected and skipped automatically.

### Rebuild timing summaries without rerunning planners

```bash
python -u patrollingAlgorithms.py segment_timing_aggregate proposed_only all
```

Aggregate-only mode searches the current working directory for `calculation_times_segment_Level-*.json` checkpoint files and rebuilds the CSV, LaTeX-table, and Markdown summaries. Run it in the directory containing the checkpoint files.

## 5. Ablation analysis

Syntax:

```text
python -u patrollingAlgorithms.py ablation [episodes] [environment]
```

Reproduce the complete ablation analysis:

```bash
python -u patrollingAlgorithms.py ablation 30 all
```

Run one level only:

```bash
python -u patrollingAlgorithms.py ablation 30 Level-4
```

The evaluated variants are:

1. full proposed method;
2. no A* initialization;
3. reduced B-Spline representation;
4. reduced-objective NSGA-II;
5. no adaptive softmin fusion; and
6. no post-processing.

The evaluation generates a JSON result file, CSV summary, LaTeX-table file, and Markdown report.

## Smoke tests versus full reproduction

Use reduced runs to check installation, multiprocessing, and output generation before starting full evaluations:

```bash
python -u patrollingAlgorithms.py segment_timing 1 Level-1 proposed_only
python -u patrollingAlgorithms.py sensitivity 1 rho
python -u patrollingAlgorithms.py ablation 1 Level-1
```

These one-episode commands are software checks. They do **not** reproduce the sample sizes, statistical summaries, or conclusions reported in the article.

Full reproduction uses:

```bash
python -u patrollingAlgorithms.py test
python -u patrollingAlgorithms.py sensitivity 30 rho_tau
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
python -u patrollingAlgorithms.py ablation 30 all
```

## Repository contents and generated outputs

The repository includes:

- the complete Python implementation; and
- publication figures derived from completed evaluations.

Raw numerical experiment outputs are not currently distributed in the repository. Running the corresponding evaluation modes generates JSON, CSV, Markdown-report, LaTeX-table, and figure files locally in the current working directory.

Publication figures are derived visual outputs and should not be interpreted as raw experimental data. Reduced one-episode commands verify software operation only and do not reproduce the full statistical results reported in the article.

## Timing interpretation

The reported timing evaluation was performed on an Intel Core i5-1155G7 processor with four physical cores and eight logical processors.

Absolute runtime depends on processor performance, operating system, package versions, background activity, and thermal conditions. Timing results should therefore be interpreted comparatively and not as platform-independent real-time guarantees.

## Troubleshooting

### `python` is not recognized

On Windows, use `py -3.12` instead of `python`. On Linux or macOS, use `python3` when required.

### `ModuleNotFoundError`

Activate the virtual environment and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

### PowerShell prevents environment activation

Allow script execution for the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, use Command Prompt and `.venv\Scripts\activate.bat`.

### A timing run was interrupted

Run the same `segment_timing` command again from the same directory. Completed checkpoint records will be skipped.

### Aggregate-only mode produces empty summaries

Confirm that the current working directory contains the `calculation_times_segment_Level-*.json` checkpoint files generated by `segment_timing`.

### Output files are not where expected

The program writes generated files to the directory from which it was launched. Check the terminal's current working directory.

### Multiprocessing messages appear out of order

The `test` and `sensitivity` modes process episodes in parallel. Interleaved progress messages are expected. Avoid launching multiple full evaluations simultaneously on the same machine.

## Citation

When using this repository, cite the associated article:

**Turan O E, Misir O, and Ozden M 2026 Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots. Measurement Science and Technology**
