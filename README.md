# Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots

This repository provides the Python implementation and evaluation data supporting the study **"Adaptive A*/NSGA-II Framework for Multi-Goal Navigation of Mobile Robots."**

The framework combines A* initialization, B-Spline trajectory generation, five-objective NSGA-II optimization, adaptive route blending, and collision-aware post-processing for sequential multi-goal mobile-robot navigation.

## Requirements

The archived experiments used:

- Python 3.12.4
- NumPy 1.26.4
- SciPy 1.14.0
- Matplotlib 3.9.0
- DEAP 1.4.3

A GPU is not required. Multi-episode evaluations use CPU multiprocessing when multiple processors are available.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OEmreTURAN/Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots.git
cd Adaptive-A-Star-and-NSGA-II-Framework-for-Multi-Goal-Navigation-of-Mobile-Robots
```

### 2. Create a virtual environment

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

If Python 3.12 is already the default interpreter, the environment can also be created with:

```bash
python -m venv .venv
```

### 3. Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import numpy, scipy, matplotlib, deap; print('Dependencies imported successfully')"
python -m py_compile patrollingAlgorithms.py
```

## Running evaluations

Run commands from the repository root using:

```text
python -u patrollingAlgorithms.py <mode> [arguments]
```

The `-u` option prints progress immediately. Generated files are written to the current working directory.

Available evaluation modes:

| Mode | Evaluation |
|---|---|
| `single` | Representative fixed-seed trajectories |
| `test` | Main 100-episode method comparison |
| `sensitivity` | Parameter sensitivity analysis |
| `segment_timing` | Route and single-segment timing analysis |
| `segment_timing_aggregate` | Summary generation from timing checkpoints |
| `ablation` | Controlled module-ablation analysis |

## 1. Representative trajectories

```bash
python -u patrollingAlgorithms.py single
```

This command uses seed 50 and evaluates all comparison methods once across the five environments. It generates trajectory-comparison figures and timing JSON files.

Use this mode first to confirm that the software runs correctly and to inspect representative paths.

## 2. Main 100-episode comparison

```bash
python -u patrollingAlgorithms.py test
```

This command evaluates all methods for 100 episodes in each of the five environments. It records:

- path length;
- maximum curvature;
- minimum obstacle clearance;
- smoothness;
- wheel effort;
- centering; and
- computation time.

It also generates JSON summaries, box plots, raincloud plots, radar charts, normalized bar charts, and metric-specific plots.

This evaluation uses CPU multiprocessing and may require substantial computation time.

## 3. Parameter sensitivity analysis

Command syntax:

```text
python -u patrollingAlgorithms.py sensitivity [episodes] [scope]
```

Available scopes:

| Scope | Evaluation |
|---|---|
| `rho_tau` | A* seed ratio and softmin temperature |
| `rho` | A* seed ratio only |
| `tau` | Softmin temperature only |
| `all` | All implemented parameter sweeps |

Run the rho and tau evaluation used in the study:

```bash
python -u patrollingAlgorithms.py sensitivity 30 rho_tau
```

Run every available parameter sweep:

```bash
python -u patrollingAlgorithms.py sensitivity 30 all
```

The `all` evaluation is considerably more expensive because every parameter value is evaluated across all five environments.

Sensitivity evaluation generates JSON and CSV summaries, LaTeX table data, an analysis report, and sensitivity figures.

These runs evaluate fixed parameter settings. They do not perform online parameter adaptation.

## 4. Route and single-segment timing

Command syntax:

```text
python -u patrollingAlgorithms.py segment_timing [episodes] [environment] [method-filter]
```

The environment argument can be `all` or one of `Level-1`, `Level-2`, `Level-3`, `Level-4`, or `Level-5`.

Available method filters:

| Filter | Evaluation |
|---|---|
| `proposed_only` | Proposed NSGA-II framework |
| `competitors_only` | Standard GA, AB-WOA-APF, and HWPSO |
| `all` | Proposed framework and all comparison methods |

Run the timing evaluation used in the study:

```bash
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
```

Run one environment only:

```bash
python -u patrollingAlgorithms.py segment_timing 100 Level-3 proposed_only
```

The timing mode records full-route runtime and individual waypoint-to-waypoint segment times. For the proposed method, one segment optimization jointly generates the Length, Smooth, Effort, Centered, Safe, and Adaptive route variants.

### Resume an interrupted timing run

Timing results are checkpointed during execution. To resume, run the same command again from the same working directory:

```bash
python -u patrollingAlgorithms.py segment_timing 100 all proposed_only
```

Completed records are detected and skipped automatically.

Timing evaluation generates environment-level checkpoint JSON files, single-segment CSV summaries, route-versus-segment CSV summaries, LaTeX table data, and a timing report.

## 5. Rebuild timing summaries without rerunning experiments

Command syntax:

```text
python -u patrollingAlgorithms.py segment_timing_aggregate [method-filter] [environment]
```

Example:

```bash
python -u patrollingAlgorithms.py segment_timing_aggregate proposed_only all
```

This mode does not run the planners. It rebuilds summary files from timing checkpoint JSON files located in the current working directory.

Run this command in the same directory that contains the checkpoint files produced by `segment_timing`.

## 6. Ablation analysis

Command syntax:

```text
python -u patrollingAlgorithms.py ablation [episodes] [environment]
```

Run the full ablation evaluation used in the study:

```bash
python -u patrollingAlgorithms.py ablation 30 all
```

Run one environment only:

```bash
python -u patrollingAlgorithms.py ablation 30 Level-4
```

The ablation evaluation compares:

1. full proposed method;
2. no A* initialization;
3. reduced B-Spline representation;
4. reduced-objective NSGA-II;
5. no adaptive softmin fusion; and
6. no post-processing.

It generates raw JSON records, CSV summaries, LaTeX table data, and an ablation report.

## Recommended evaluation sequence

For first-time use:

1. Create and activate the virtual environment.
2. Install `requirements.txt`.
3. Verify the installation.
4. Run the representative `single` evaluation.
5. Run small evaluation jobs to confirm multiprocessing and output generation.
6. Start the complete evaluations only when sufficient CPU time is available.

Small evaluation examples:

```bash
python -u patrollingAlgorithms.py sensitivity 1 rho
python -u patrollingAlgorithms.py segment_timing 1 Level-1 proposed_only
python -u patrollingAlgorithms.py ablation 1 Level-1
```

These reduced runs are software checks and should not be compared directly with the completed study results.

## Completed evaluation data

Completed JSON, CSV, and figure outputs are included in the repository. They can be inspected without rerunning the full experiments.

The timing, sensitivity, and ablation files contain the deterministic seeds used for those evaluations. The archived main-comparison files contain the complete 100-episode metric values used for the statistical summaries.

## Timing interpretation

The archived timing experiments were performed on an Intel Core i5-1155G7 processor with four physical cores and eight logical processors.

Runtime depends on the processor, operating system, installed package versions, background processes, and thermal conditions. The reported values should therefore be interpreted comparatively and not as platform-independent real-time guarantees.

## Troubleshooting

### `python` is not recognized

On Windows, use `py -3.12` instead of `python`. On Linux or macOS, use `python3` if required.

### `ModuleNotFoundError`

Activate the virtual environment and reinstall the dependencies:

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

Execute the same `segment_timing` command from the same directory. Completed checkpoint records will be skipped.

### Output files are not visible in the repository root

The program writes files to the directory from which it was launched. Check the current working directory used by the terminal.

### Multiprocessing messages appear out of order

The `test` and `sensitivity` modes process episodes in parallel. Interleaved terminal progress messages are expected.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated article when using the software or evaluation data.
