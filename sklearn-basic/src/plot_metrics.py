from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from expops.reporting import chart


def _series(metric: Any) -> Tuple[list[int], list[float]]:
    """Convert {"1": 0.1, "2": 0.2, ...} into ([1,2], [0.1,0.2])."""
    if not isinstance(metric, dict):
        return ([], [])
    pts: list[tuple[int, float]] = []
    for k, v in metric.items():
        try:
            step = int(k)
            val = float(v)
        except Exception:
            continue
        pts.append((step, val))
    pts.sort(key=lambda x: x[0])
    return ([p[0] for p in pts], [p[1] for p in pts])


def _last(metric: Any) -> float | None:
    xs, ys = _series(metric)
    return ys[-1] if ys else None


@chart()
def plot_metrics(metrics: Dict[str, Any]) -> None:
    """
    Chart entrypoint invoked by ExpOps chart runner.
    """
    import matplotlib.pyplot as plt

    train_block = metrics.get("train") if isinstance(metrics, dict) else None
    eval_block = metrics.get("eval") if isinstance(metrics, dict) else None

    train_block = train_block if isinstance(train_block, dict) else {}
    eval_block = eval_block if isinstance(eval_block, dict) else {}

    # Accuracy bars (final train vs eval)
    train_acc = _last(train_block.get("accuracy"))
    eval_acc = _last(eval_block.get("accuracy"))
    labels: list[str] = []
    values: list[float] = []
    if train_acc is not None:
        labels.append("train")
        values.append(float(train_acc))
    if eval_acc is not None:
        labels.append("eval")
        values.append(float(eval_acc))

    fig, ax = plt.subplots(figsize=(4, 3))
    if values:
        ax.bar(labels, values)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("accuracy")
    else:
        ax.text(0.5, 0.5, "No accuracy metrics found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Accuracy")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    plt.savefig("accuracy.png", dpi=150)
    plt.close(fig)

