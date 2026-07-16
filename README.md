# Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots

Python implementation of a multi-objective path-planning framework for sequential mobile-robot navigation. The method combines A* initialization, B-Spline trajectory representation, NSGA-II optimization, adaptive route blending, and collision-aware post-processing across five simulation environments (`Level-1` through `Level-5`).

- **Authors:** Osman Emre Turan, Oğuz Mısır, and Mustafa Özden

## Main capabilities

- Representative trajectory generation for the proposed framework and comparison methods
- Multi-episode evaluation of path length, curvature, clearance, smoothness, wheel effort, centering, and runtime
- Fixed-parameter sensitivity analysis
- Route-level and waypoint-to-waypoint segment timing
- Checkpointed timing runs with resume support
- Controlled ablation analysis of the framework components
- Publication figures for trajectory, distribution, radar, and sensitivity analyses

## Requirements

The reference software environment uses:

- Python 3.12.4
- NumPy 1.26.4
- SciPy 1.14.0
- Matplotlib 3.9.0
- DEAP 1.4.3

A GPU is not required. Multi-episode modes use CPU multiprocessing when multiple logical processors are available.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/OEmreTURAN/Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots.git
cd Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots
```

Create a virtual environment.

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

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import numpy, scipy, matplotlib, deap; print('Dependencies imported successfully')"
python -m py_compile patrollingAlgorithms.py
```

## Quick start

Run a one-episode check of the proposed planner in `Level-1`:

```bash
python -u patrollingAlgorithms.py segment_timing 1 Level-1 proposed_only
```

This command initializes the planner, evaluates one complete route, records its waypoint-to-waypoint segment times, and writes the following files to the current working directory:

```text
calculation_times_segment_Level-1.json
single_segment_time_summary.csv
route_vs_segment_runtime_summary.csv
single_segment_time_for_latex.tex
segment_timing_instrumentation_report.md
```

A one-episode run is intended to verify installation and output generation. Use larger episode counts for statistical analysis.

## Command overview

```text
python -u patrollingAlgorithms.py <mode> [arguments]
```

The `-u` option displays progress immediately. Generated numerical outputs, reports, tables, and new figures are written to the current working directory.

| Mode | Purpose | Typical command |
|---|---|---|
| `single` | Generate representative trajectories | `python -u patrollingAlgorithms.py single` |
| `test` | Compare all methods over 100 episodes | `python -u patrollingAlgorithms.py test` |
| `sensitivity` | Evaluate parameter sensitivity | `python -u patrollingAlgorithms.py sensitivity 30 rho_tau` |
| `segment_timing` | Measure route and segment planning time | `python -u patrollingAlgorithms.py segment_timing 100 all proposed_only` |
| `segment_timing_aggregate` | Rebuild summaries from timing checkpoints | `python -u patrollingAlgorithms.py segment_timing_aggregate proposed_only all` |
| `ablation` | Quantify component contributions | `python -u patrollingAlgorithms.py ablation 30 all` |

## Representative trajectories

```bash
python -u patrollingAlgorithms.py single
```

**Purpose:** Visually inspect the paths produced by the proposed framework and comparison methods under a common fixed seed.

The command uses seed 50 and evaluates all five environments. It generates:

```text
Patrolling_9Paths_*.png
calculation_times_single_*.json
```

The trajectory figures overlay Standard GA, AB-WOA-APF, HWPSO, five NSGA-II route variants, and the Adaptive route.

## Multi-episode method comparison

```bash
python -u patrollingAlgorithms.py test
```

**Purpose:** Compare path quality, variability, and computation time across all methods and environments.

The command runs 100 episodes per environment and evaluates:

- path length;
- maximum curvature;
- minimum obstacle clearance;
- smoothness;
- wheel effort;
- centering; and
- computation time.

Generated outputs include:

```text
all_metrics_*.json
distances_*.json
calculation_times_*.json
Boxplot_*.png
Raincloud_*.png
Radar_*.png
NormalizedBar_*.png
```

Metric-specific box plots are also generated. This mode uses multiprocessing and can require substantial CPU time.

## Parameter sensitivity analysis

```text
python -u patrollingAlgorithms.py sensitivity [episodes] [scope]
```

**Purpose:** Measure how fixed algorithm parameters affect path quality, safety, and computation time.

Available scopes:

| Scope | Parameters evaluated |
|---|---|
| `rho_tau` | A* seed ratio and softmin temperature |
| `rho` | A* seed ratio only |
| `tau` | Softmin temperature only |
| `all` | All implemented parameter sweeps |

Evaluate the A* seed ratio and softmin temperature with 30 episodes per setting and environment:

```bash
python -u patrollingAlgorithms.py sensitivity 30 rho_tau
```

Evaluate all implemented parameter sweeps:

```bash
python -u patrollingAlgorithms.py sensitivity 30 all
```

Generated outputs include:

```text
sensitivity_results.json
sensitivity_rho_summary.csv
sensitivity_tau_summary.csv
sensitivity_rho_tau_for_latex.tex
sensitivity_rho_tau_report.md
Sensitivity_*.png
```

This mode evaluates fixed parameter settings. It does not perform online parameter adaptation.

## Route and single-segment timing

```text
python -u patrollingAlgorithms.py segment_timing [episodes] [environment] [method-filter]
```

**Purpose:** Measure complete-route runtime and individual waypoint-to-waypoint planning times using the same instrumented run.

The environment argument accepts `all` or `Level-1` through `Level-5`.

Available method filters:

| Filter | Methods evaluated |
|---|---|
| `proposed_only` | Proposed NSGA-II framework |
| `competitors_only` | Standard GA, AB-WOA-APF, and HWPSO |
| `all` | Proposed framework and all comparison methods |

Evaluate the proposed framework for 100 episodes in every environment:

```bash
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
```

Evaluate one environment:

```bash
python -u patrollingAlgorithms.py segment_timing 100 Level-3 proposed_only
```

One proposed-method segment optimization jointly generates the Length, Smooth, Effort, Centered, Safe, and Adaptive route variants.

Generated outputs include:

```text
calculation_times_segment_Level-1.json
...
calculation_times_segment_Level-5.json
single_segment_time_summary.csv
route_vs_segment_runtime_summary.csv
single_segment_time_for_latex.tex
segment_timing_instrumentation_report.md
```

### Resume a timing run

Timing checkpoints are updated during execution. After an interruption, run the same command again from the same directory:

```bash
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
```

Completed records are detected and skipped automatically.

### Rebuild timing summaries

```bash
python -u patrollingAlgorithms.py segment_timing_aggregate proposed_only all
```

**Purpose:** Recreate CSV, LaTeX-table, and Markdown summaries without executing the planners again.

Run aggregate-only mode from the directory containing the `calculation_times_segment_Level-*.json` checkpoint files.

## Ablation analysis

```text
python -u patrollingAlgorithms.py ablation [episodes] [environment]
```

**Purpose:** Quantify how individual framework components affect path quality, safety, and runtime.

Evaluate all five environments with 30 episodes each:

```bash
python -u patrollingAlgorithms.py ablation 30 all
```

Evaluate one environment:

```bash
python -u patrollingAlgorithms.py ablation 30 Level-4
```

The implemented variants are:

1. full proposed method;
2. no A* initialization;
3. reduced B-Spline representation;
4. reduced-objective NSGA-II;
5. no adaptive softmin fusion; and
6. no post-processing.

Generated outputs include:

```text
ablation_results_by_environment.json
ablation_summary_all.csv
ablation_summary_for_latex.tex
ablation_experiment_report.md
```

## Small validation runs

Use reduced episode counts to verify the software before scheduling longer analyses:

```bash
python -u patrollingAlgorithms.py segment_timing 1 Level-1 proposed_only
python -u patrollingAlgorithms.py sensitivity 1 rho
python -u patrollingAlgorithms.py ablation 1 Level-1
```

These commands check execution and file generation. Their sample sizes are not intended for statistical interpretation.

## Outputs and publication figures

Numerical outputs are generated locally when commands are executed. The repository does not distribute episode-level JSON or CSV records.

Derived publication figures are provided under:

```text
figures/paths/
figures/boxplots/
figures/raincloud/
figures/radar/
figures/sensitivity/
```

Run evaluations from a dedicated directory when you want to keep newly generated outputs separate from the repository files.

## Timing interpretation

The reference timing evaluation was conducted on an Intel Core i5-1155G7 processor with four physical cores and eight logical processors.

Absolute runtime depends on the processor, operating system, package versions, background activity, and thermal conditions. Timing measurements should therefore be interpreted comparatively rather than as platform-independent real-time guarantees.

## Troubleshooting

### `python` is not recognized

On Windows, use `py -3.12` instead of `python`. On Linux or macOS, use `python3` when required.

### `ModuleNotFoundError`

Activate the virtual environment and install the dependencies:

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

Run the same `segment_timing` command again from the same directory. Completed records will be skipped.

### Aggregate-only mode produces empty summaries

Confirm that the current working directory contains the checkpoint JSON files created by `segment_timing`.

### Output files are not where expected

The program writes outputs to the terminal's current working directory.

### Multiprocessing messages appear out of order

The `test` and `sensitivity` modes process episodes in parallel. Interleaved progress messages are expected. Avoid launching multiple long analyses simultaneously on the same machine.

## Citation

If you use this software in academic work, cite the accompanying manuscript once it has been published:

**Turan O E, Mısır O and Ozden M, Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots.**

## Support

- Contact [emre.turan@btu.edu.tr](mailto:emre.turan@btu.edu.tr) for questions about the methodology or manuscript.
