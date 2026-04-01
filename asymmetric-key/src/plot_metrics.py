from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expops.reporting import chart


def _get_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and v:
        items = sorted(v.items(), key=lambda kv: int(str(kv[0])) if str(kv[0]).isdigit() else 0)
        try:
            return float(items[-1][1]) if items else None
        except Exception:
            return None
    return None


def _label_from_probe_key(k: str) -> str:
    s = str(k)
    try:
        m = re.search(r"process\[@name='([^']+)'\]", s)
    except Exception:
        m = None
    if m and m.group(1):
        return m.group(1)
    return s.split("/")[-1]


def _parse_rsa_cell_label(label: str) -> Optional[Tuple[int, int]]:
    """Parse probe short keys like rsa_768_p2 -> (768, 2)."""
    s = str(label).strip()
    if s.endswith("_bench"):
        s = s[: -len("_bench")]
    m = re.match(r"^rsa_(\d+)_p(\d+)$", s, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


@chart()
def asymmetric_summary_chart(metrics: Dict[str, Any]) -> None:
    """
    Heatmap of mean RSA encryption time (ms): rows = key size (768, 1024, 2048),
    columns = payload bytes (2, 3, 4). Expects each bench to log mean_encrypt_ms.
    """
    key_sizes = (768, 1024, 2048)
    payload_sizes = (2, 3, 4)
    grid: Dict[Tuple[int, int], float] = {}

    for k, m in (metrics or {}).items():
        if not isinstance(m, dict):
            continue
        label = _label_from_probe_key(str(k))
        if label.endswith("_bench"):
            continue
        parsed = _parse_rsa_cell_label(label)
        if parsed is None:
            continue
        ks, ps = parsed
        if ks not in key_sizes or ps not in payload_sizes:
            continue
        v = _get_float(m.get("mean_encrypt_ms"))
        if v is None:
            continue
        grid[(ks, ps)] = float(v)

    if not grid:
        return

    mat = np.full((len(key_sizes), len(payload_sizes)), np.nan, dtype=float)
    for i, ks in enumerate(key_sizes):
        for j, ps in enumerate(payload_sizes):
            mat[i, j] = grid.get((ks, ps), np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(payload_sizes)), labels=[f"{p} B" for p in payload_sizes])
    ax.set_yticks(np.arange(len(key_sizes)), labels=[f"{ks} bit" for ks in key_sizes])
    ax.set_xlabel("Payload size")
    ax.set_ylabel("RSA key size")
    ax.set_title("Mean RSA encrypt time (OAEP-SHA256, ms)\n768 bit replaces 512 for PKCS#1 OAEP limits")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean_encrypt_ms")

    for i in range(len(key_sizes)):
        for j in range(len(payload_sizes)):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val >= np.nanmax(mat) * 0.5 else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig("asymmetric_summary_chart.png", dpi=160)
    plt.close(fig)
