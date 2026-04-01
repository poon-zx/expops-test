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
    """Parse probe short keys like rsa_1024_p2 -> (1024, 2)."""
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
    Grouped bar chart of mean RSA encrypt vs decrypt time (ms) per process.

    Expects each RSA bench process (e.g. rsa_1024_p2) to log:
    - mean_encrypt_ms
    - mean_decrypt_ms
    """
    rsa_rows: list[Tuple[Tuple[int, int], str, float, float]] = []

    for k, m in (metrics or {}).items():
        if not isinstance(m, dict):
            continue
        label = _label_from_probe_key(str(k))
        if label.endswith("_bench"):
            continue
        parsed = _parse_rsa_cell_label(label)
        if parsed is None:
            continue

        enc = _get_float(m.get("mean_encrypt_ms"))
        dec = _get_float(m.get("mean_decrypt_ms"))
        if enc is None or dec is None:
            continue
        rsa_rows.append((parsed, label, float(enc), float(dec)))

    if not rsa_rows:
        return

    rsa_rows.sort(key=lambda t: (t[0][0], t[0][1]))  # (key_size, payload)
    labels = [t[1] for t in rsa_rows]
    encrypt_ms = np.asarray([t[2] for t in rsa_rows], dtype=float)
    decrypt_ms = np.asarray([t[3] for t in rsa_rows], dtype=float)

    x = np.arange(len(labels), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.bar(x - width / 2.0, encrypt_ms, width=width, label="Encrypt")
    ax.bar(x + width / 2.0, decrypt_ms, width=width, label="Decrypt")

    ax.set_xticks(x, labels=labels, rotation=20, ha="right")
    ax.set_ylabel("Mean time (ms)")
    ax.set_title("Mean RSA encrypt vs decrypt time (OAEP-SHA256, ms)\nNote: this runtime enforces RSA key_size ≥ 1024")
    ax.legend(loc="best")

    ymax = float(np.nanmax(np.concatenate([encrypt_ms, decrypt_ms])))
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0.0, ymax * 1.18)

    fig.tight_layout()
    plt.savefig("asymmetric_summary_chart.png", dpi=160)
    plt.close(fig)
