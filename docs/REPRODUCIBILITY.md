# Reproducibility Notes

## Software environment

- Python 3.12.4
- NumPy 1.26.4
- SciPy 1.14.0
- Matplotlib 3.9.0
- DEAP 1.4.3

Install the pinned packages with `python -m pip install -r requirements.txt`.

## Experiment modes

| Analysis | Command | Repetitions |
|---|---|---:|
| Representative trajectories | `python -u patrollingAlgorithms.py single` | Seed 50 |
| Main comparison | `python -u patrollingAlgorithms.py test` | 100 per environment |
| rho/tau sensitivity | `python -u patrollingAlgorithms.py sensitivity 30 rho_tau` | 30 per setting/environment |
| Segment timing | `python -u patrollingAlgorithms.py segment_timing 100 all proposed_only` | 100 per environment |
| Ablation | `python -u patrollingAlgorithms.py ablation 30 all` | 30 per environment |

The sensitivity, segment-timing, and ablation JSON/CSV records contain the deterministic seeds used in those analyses. Segment timing supports checkpointing and resume; aggregate-only mode rebuilds summaries without rerunning optimization.

The archived main-comparison JSON files contain the complete 100-episode metric values used for the reported statistics. The legacy main-comparison files did not store their individual seed identifiers, so the archived values, rather than an exact replay of that historical random sequence, are the authoritative evidence for those tables and plots.

## Environment naming

Always use the mapping in `data/environment_mapping.json`. Level-3 is Moderate-III and Level-4 is Moderate-II. Internal identifiers are retained only where useful for tracing the procedural environment definitions in the source code.

## Timing interpretation

Timing data were collected on an Intel Core i5-1155G7 processor with four physical cores and eight logical processors. Absolute runtimes depend on hardware, operating system, background load, and package versions. The stored values support comparative computational-cost analysis, not a platform-independent real-time guarantee.

## Scope limitations

The experiments use static procedural maps, fixed waypoint order, and segment-wise optimization. The source does not implement dynamic-obstacle replanning or adaptive online tuning of rho and tau.
