# Asymmetric Key Operations (ExpOps Demo)

This demo benchmarks asymmetric cryptography operations in parallel branches to showcase ExpOps’ process graph execution, declarative parameter sweeps, caching, and reporting (dynamic + static charts).

## What it demonstrates

- **Parallelism**: RSA / ECDSA / Ed25519 branches run independently.
- **Hyperparameter sweeps**: RSA key size is configured declaratively (same code, different `key_size`).
- **Caching**: RSA-4096 key generation is intentionally slow enough to show a dramatic cache hit on re-run.
- **Reporting**:
  - **Dynamic**: live key-generation latency per trial for a few selected branches.
  - **Static**: post-run bar chart comparing mean keygen/sign/verify across all branches.

## Project layout

- `configs/project_config.yaml`: pipeline graph + processes + chart processes
- `configs/compute_config.yaml`: compute backend settings
- `src/asymmetric_key_model.py`: benchmark processes (ExpOps `@process()` + `@step()`)
- `src/plot_metrics.js`: dynamic Chart.js chart
- `src/plot_metrics.py`: static matplotlib chart

## How to run

Run this project the same way you run other `mlops-platform/*` demos (e.g. `premier-league`) using your ExpOps CLI / UI workflow.

Suggested live demo flow for an evaluator:

1. Run once and open the **dynamic chart** (`keygen_live`) while the pipeline is executing.
2. After completion, view `asymmetric_summary_chart.png`.
3. Run again with the same config to show **cache hits** (second run should be dramatically faster, especially RSA-4096 keygen).

