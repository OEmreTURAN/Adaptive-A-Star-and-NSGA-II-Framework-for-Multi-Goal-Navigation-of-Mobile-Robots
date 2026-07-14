# Data Dictionary

## Authoritative environment mapping

| Manuscript environment | Internal identifier |
|---|---|
| Level-1 | Easy |
| Level-2 | Moderate-I |
| Level-3 | Moderate-III |
| Level-4 | Moderate-II |
| Level-5 | Hard |

`data/environment_mapping.json` provides the same mapping in machine-readable form.

## Path metrics

`data/path_metrics/distances_Level-*.json` contains per-method path-length summary statistics and the underlying 100 episode values. `all_metrics_Level-*.json` contains path length, maximum curvature, minimum clearance, smoothness, wheel effort, and centering measurements used for the multi-metric analysis.

## Runtime

`data/runtime/raw_json/calculation_times_Level-*.json` contains the stored full-route environment summaries. The source files originally used internal environment filenames and were renamed according to the authoritative mapping above.

`data/runtime/raw_json/calculation_times_segment_Level-*.json` contains 100 proposed-method timing episodes per environment, including route summaries and individual waypoint-to-waypoint segment records. One NSGA-II segment optimization jointly generates the Length, Smooth, Effort, Centered, Safe, and Adaptive route variants.

The Level-3 and Level-4 segment files contain Moderate-III and Moderate-II measurements, respectively. Some route-summary objects retained stale manuscript labels from the earlier naming convention. In these public copies, only label and episode-ID fields were normalized from the trusted `internal_environment` field. Timing measurements and seeds were preserved exactly.

The CSV and JSON files directly under `data/runtime/` are derived summaries used for the runtime tables.

## Ablation

`ablation_results_by_environment.json` contains records for six variants across all five environments. `ablation_summary_all.csv` contains the corresponding environment-level summaries. Environment rows include both manuscript and internal identifiers.

## Sensitivity

`sensitivity_results.json` contains the rho and tau sweep results. Its public top-level environment keys use Level-1 through Level-5. The CSV files include both manuscript and internal identifiers for traceability.

## Figures

The `figures/` directories contain manuscript-facing PNG outputs. Their Level-3 and Level-4 filenames follow the authoritative mapping above.
