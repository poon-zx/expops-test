# Asymmetric Key — RSA encryption benchmark (paper-style)

This demo benchmarks **RSA encryption and decryption** latency in an experiment matrix inspired by *Cryptographic Algorithms Benchmarking: A Case Study* (Boicea et al.): RSA modulus sizes × short plaintext sizes, with **10 trials per cell** and **mean encryption/decryption time in milliseconds** reported per configuration.

## Experiment design

- **Key sizes (bits):** 1024, 2048 — crossed with **payload sizes (bytes):** 2, 3, 4 (short messages analogous to small bit-string lengths in the paper).
- **Six process runs:** one `define_rsa_bench` process per (key size, payload size) pair, each fed by a dedicated `data_generation_*` branch so payloads match the column.
- **Cryptography:** RSA encryption/decryption with **OAEP** and **SHA-256** (`cryptography`); each trial generates a fresh key pair (**key generation is not timed or logged**), times **encrypt** and **decrypt**, and verifies correctness (decrypt output must match plaintext).
- **Metrics:** `encrypt_ms` and `decrypt_ms` are logged per trial (step-indexed), and `mean_encrypt_ms` / `mean_decrypt_ms` are logged once per configuration (mean over 10 trials).

### Why not 512-bit RSA?

PKCS#1 OAEP with SHA-256 needs enough modulus length to fit the padding overhead. A **512-bit** modulus is **too small** for OAEP-SHA256 (no room for even a 1-byte message). Additionally, the `cryptography` backend used here enforces **RSA key_size ≥ 1024**, so this project uses 1024 and 2048.

## Project layout

- `configs/project_config.yaml` — pipeline: three payload generators, six RSA benches, one static chart, and one dynamic chart for `p2` trial series comparison.
- `configs/compute_config.yaml` — compute backend settings.
- `src/asymmetric_key_model.py` — `data_generation`, `define_rsa_bench` (plus unused-in-pipeline ECDSA/Ed25519 helpers).
- `src/plot_metrics.py` — grouped bar chart `asymmetric_summary_chart.png` (encrypt vs decrypt per RSA process).
- `src/plot_metrics.js` — dynamic chart `rsa_p2_timing_comparison` (trial-by-trial `rsa_1024_p2` vs `rsa_2048_p2`, encrypt/decrypt).

## How to run

Run this project the same way as other `mlops-platform/*` demos using your ExpOps CLI or UI. After a run, open **`asymmetric_summary_chart.png`** for the grouped bar chart of mean encrypt vs decrypt time per RSA process, and view **`rsa_p2_timing_comparison`** in the dynamic charts UI for trial-level timing comparison.

## Caching

Re-runs can still show cache benefits on the RSA branches (especially larger keys), depending on your ExpOps cache backend configuration.
