from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
        # Step-indexed values: {"1": x, "2": y, ...} or similar
        items = sorted(v.items(), key=lambda kv: int(str(kv[0])) if str(kv[0]).isdigit() else 0)
        try:
            return float(items[-1][1]) if items else None
        except Exception:
            return None
    return None


def _label_from_probe_key(k: str) -> str:
    s = str(k)
    # Prefer the leaf process name if it's a resolved canonical key
    m = None
    try:
        import re

        m = re.search(r"process\[@name='([^']+)'\]", s)
    except Exception:
        m = None
    if m and m.group(1):
        return m.group(1)
    return s.split("/")[-1]


@chart()
def asymmetric_summary_chart(metrics: Dict[str, Any]) -> None:
    """
    Static chart: mean keygen/sign/verify (ms) across all asymmetric branches.
    Expects each bench process to log:
      - mean_keygen_ms
      - mean_sign_ms
      - mean_verify_ms
    """

    # metrics is keyed by probe_paths keys when one match, or canonical resolved paths when many
    rows: List[Tuple[str, float, float, float]] = []
    for k, m in (metrics or {}).items():
        if not isinstance(m, dict):
            continue
        mk = _get_float(m.get("mean_keygen_ms"))
        ms = _get_float(m.get("mean_sign_ms"))
        mv = _get_float(m.get("mean_verify_ms"))
        if mk is None and ms is None and mv is None:
            continue
        rows.append((_label_from_probe_key(str(k)), float(mk or 0.0), float(ms or 0.0), float(mv or 0.0)))

    if not rows:
        return

    # Stable order for readability (and for the professor’s “RSA scaling” story)
    desired = [
        "rsa_1024_bench",
        "rsa_2048_bench",
        "rsa_4096_bench",
        "ecdsa_p256_bench",
        "ecdsa_p384_bench",
        "ed25519_bench",
    ]
    order = {name: i for i, name in enumerate(desired)}
    rows.sort(key=lambda r: order.get(r[0], 10_000))

    labels = [r[0].replace("_", " ").upper() for r in rows]
    keygen = [r[1] for r in rows]
    sign = [r[2] for r in rows]
    verify = [r[3] for r in rows]

    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(13, 4.2))
    b1 = ax.bar(x - width, keygen, width, label="Keygen (mean ms)", color="steelblue")
    b2 = ax.bar(x, sign, width, label="Sign (mean ms)", color="mediumseagreen")
    b3 = ax.bar(x + width, verify, width, label="Verify (mean ms)", color="coral")

    ax.set_ylabel("Milliseconds (mean)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    # Log scale helps show Ed25519/ECDSA when RSA-4096 dominates
    try:
        ax.set_yscale("log")
    except Exception:
        pass

    for rect in list(b1) + list(b2) + list(b3):
        h = rect.get_height()
        if not np.isfinite(h) or h <= 0:
            continue
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    plt.savefig("asymmetric_summary_chart.png", dpi=160)
    plt.close(fig)

