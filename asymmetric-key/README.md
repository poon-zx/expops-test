# Asymmetric Key — RSA encryption benchmark (paper-style)

This demo benchmarks **RSA encryption** latency in a **3×3 experiment matrix** inspired by *Cryptographic Algorithms Benchmarking: A Case Study* (Boicea et al.): three RSA modulus sizes × three short plaintext sizes, with **10 trials per cell** and **mean encryption time in milliseconds** as the only reported metric.

## Experiment design

- **Key sizes (bits):** 768, 1024, 2048 — crossed with **payload sizes (bytes):** 2, 3, 4 (short messages analogous to small bit-string lengths in the paper).
- **Nine process runs:** one `define_rsa_bench` process per (key size, payload size) pair, each fed by a dedicated `data_generation_*` branch so payloads match the column.
- **Cryptography:** RSA **encryption** with **OAEP** and **SHA-256** (`cryptography`); each trial generates a fresh key pair (**key generation is not timed or logged**), times **encrypt** only, then **decrypts** to verify correctness (decrypt not timed).
- **Metric:** `mean_encrypt_ms` — logged once per configuration (mean over 10 trials).

### Why 768 bits instead of 512?

PKCS#1 OAEP with SHA-256 needs enough modulus length to fit the padding overhead. A **512-bit** modulus is **too small** for OAEP-SHA256 (no room for even a 1-byte message). The smallest tier here is **768 bits** so all cells are valid; this is called out in the project description and on the chart subtitle.

## Project layout

- `configs/project_config.yaml` — pipeline: three payload generators, nine RSA benches, one static chart.
- `configs/compute_config.yaml` — compute backend settings.
- `src/asymmetric_key_model.py` — `data_generation`, `define_rsa_bench` (plus unused-in-pipeline ECDSA/Ed25519 helpers).
- `src/plot_metrics.py` — static heatmap `asymmetric_summary_chart.png`.
- `src/plot_metrics.js` — empty module (no dynamic charts).

## How to run

Run this project the same way as other `mlops-platform/*` demos using your ExpOps CLI or UI. After a run, open **`asymmetric_summary_chart.png`** for the 3×3 mean encrypt time heatmap.

## Caching

Re-runs can still show cache benefits on the nine RSA branches (especially larger keys), depending on your ExpOps cache backend configuration.
