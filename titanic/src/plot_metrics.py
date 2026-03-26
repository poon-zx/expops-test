from __future__ import annotations

from typing import Any, Dict, Optional
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expops.reporting import chart

logger = logging.getLogger(__name__)


def _get_float(v: Any) -> Optional[float]:
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


@chart()
def metrics_comparison(metrics: Dict[str, Any]) -> None:
    """
    Static chart comparing classification metrics across models.
    Expects probe_paths keys: linear, nn_a, nn_b; each logging test_accuracy/test_precision/test_f1.
    """

    groups = {
        "Linear": metrics.get("linear", {}) or {},
        "NN A": metrics.get("nn_a", {}) or {},
        "NN B": metrics.get("nn_b", {}) or {},
    }

    rows = []
    for name, m in groups.items():
        if not isinstance(m, dict):
            continue
        acc = _get_float(m.get("test_accuracy"))
        prec = _get_float(m.get("test_precision"))
        f1 = _get_float(m.get("test_f1"))
        if acc is None and prec is None and f1 is None:
            continue
        rows.append((name, float(acc or 0.0), float(prec or 0.0), float(f1 or 0.0)))

    if not rows:
        logger.warning("metrics_comparison: no metrics found. metric_keys=%s", sorted((metrics or {}).keys()))
        return

    labels = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    precs = [r[2] for r in rows]
    f1s = [r[3] for r in rows]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))

    b1 = ax.bar(x - width, accs, width, label="Accuracy", color="steelblue")
    b2 = ax.bar(x, precs, width, label="Precision", color="mediumseagreen")
    b3 = ax.bar(x + width, f1s, width, label="F1", color="coral")

    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    for rect in list(b1) + list(b2) + list(b3):
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    plt.savefig("metrics_comparison.png", dpi=160)
    plt.close(fig)

