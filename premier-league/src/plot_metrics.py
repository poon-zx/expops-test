from __future__ import annotations

from typing import Dict, Any, Optional
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expops.reporting import chart, ChartContext

logger = logging.getLogger(__name__)


@chart()
def pca_scree(metrics: Dict[str, Any]) -> None:
    """
    Static chart showing PCA explained variance ratio and cumulative variance.
    Expects metrics from feature_engineering/encode_and_pca step containing:
      - pca_explained_variance_ratio: list[float]
      - pca_cumulative_variance: list[float]
    """

    evr = metrics.get("feat", {}).get("pca_explained_variance_ratio", [])
    cum = metrics.get("feat", {}).get("pca_cumulative_variance", [])

    if not isinstance(evr, (list, tuple)) or len(evr) == 0:
        return

    xs = np.arange(1, len(evr) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(xs, evr, color="steelblue", alpha=0.7, label="Explained Variance Ratio")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    if isinstance(cum, (list, tuple)) and len(cum) == len(evr):
        ax2.plot(xs, cum, color="coral", marker="o", label="Cumulative Variance")
        ax2.set_ylabel("Cumulative Variance")

    fig.tight_layout()
    plt.savefig("pca_scree.png", dpi=150)
    plt.close(fig)


@chart()
def goals_distribution(metrics: Dict[str, Any]) -> None:
    """
    Static chart showing histograms of home and away goals.
    Expects metrics from feature_engineering/encode_and_pca step containing:
      - goals_hist_home: dict[str, int]
      - goals_hist_away: dict[str, int]
    """

    feat_metrics = metrics.get("feat", {})
    g_home = feat_metrics.get("goals_hist_home", {}) or {}
    g_away = feat_metrics.get("goals_hist_away", {}) or {}

    if not g_home and not g_away:
        return

    keys = sorted(set([int(k) for k in g_home.keys()] + [int(k) for k in g_away.keys()]))
    vals_home = [int(g_home.get(str(k), 0)) for k in keys]
    vals_away = [int(g_away.get(str(k), 0)) for k in keys]

    x = np.arange(len(keys))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, vals_home, width, label="Home Goals", color="slateblue")
    ax.bar(x + width / 2, vals_away, width, label="Away Goals", color="seagreen")
    ax.set_xticks(x, [str(k) for k in keys])
    ax.set_xlabel("Goals")
    ax.set_ylabel("Count")
    ax.set_title("Goals Distribution (Home vs Away)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig("goals_distribution.png", dpi=150)
    plt.close(fig)


@chart()
def test_metrics_comparison(metrics: Dict[str, Any]) -> None:
    """
    Static chart comparing classification metrics across baseline/best models and ensemble.
    Expected keys: linear, nn_best, xgb_best, ensemble; each with test_accuracy, test_precision, test_f1.
    """

    def get_value(data):
        if isinstance(data, dict) and data:
            items = sorted(data.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
            return float(items[-1][1]) if items else None
        if isinstance(data, (int, float)):
            return float(data)
        return None

    def collect_by_prefix(prefix: str) -> Dict[str, Any]:
        return {k: v for k, v in (metrics or {}).items() if str(k).startswith(prefix)}

    def aggregate_metric(group_metrics: Dict[str, Any], metric_key: str) -> Optional[float]:
        values = []
        for m in (group_metrics or {}).values():
            if not isinstance(m, dict):
                continue
            val = get_value(m.get(metric_key))
            if val is not None:
                values.append(val)
        if not values:
            return None
        return float(np.mean(values))

    legacy_keys = {"linear", "nn_best", "xgb_best", "ensemble"}
    if any(k in metrics for k in legacy_keys):
        groups = {
            "Linear": metrics.get("linear", {}),
            "NN (Best)": metrics.get("nn_best", {}),
            "XGB (Best)": metrics.get("xgb_best", {}),
            "Ensemble": metrics.get("ensemble", {}),
        }
        group_values = {}
        for label, m in groups.items():
            if not isinstance(m, dict):
                continue
            group_values[label] = {
                "test_accuracy": get_value(m.get("test_accuracy")),
                "test_precision": get_value(m.get("test_precision")),
                "test_f1": get_value(m.get("test_f1")),
            }
    else:
        group_values = {
            "Linear": {
                "test_accuracy": aggregate_metric(collect_by_prefix("linear_inference"), "test_accuracy"),
                "test_precision": aggregate_metric(collect_by_prefix("linear_inference"), "test_precision"),
                "test_f1": aggregate_metric(collect_by_prefix("linear_inference"), "test_f1"),
            },
            "NN (Best)": {
                "test_accuracy": aggregate_metric(collect_by_prefix("nn_best_inference"), "test_accuracy"),
                "test_precision": aggregate_metric(collect_by_prefix("nn_best_inference"), "test_precision"),
                "test_f1": aggregate_metric(collect_by_prefix("nn_best_inference"), "test_f1"),
            },
            "XGB (Best)": {
                "test_accuracy": aggregate_metric(collect_by_prefix("xgb_best_inference"), "test_accuracy"),
                "test_precision": aggregate_metric(collect_by_prefix("xgb_best_inference"), "test_precision"),
                "test_f1": aggregate_metric(collect_by_prefix("xgb_best_inference"), "test_f1"),
            },
            "Ensemble": {
                "test_accuracy": aggregate_metric(collect_by_prefix("ensemble_inference"), "test_accuracy"),
                "test_precision": aggregate_metric(collect_by_prefix("ensemble_inference"), "test_precision"),
                "test_f1": aggregate_metric(collect_by_prefix("ensemble_inference"), "test_f1"),
            },
        }

    labels = []
    accs = []
    precs = []
    f1s = []
    for label, m in group_values.items():
        acc = m.get("test_accuracy")
        prec = m.get("test_precision")
        f1 = m.get("test_f1")
        if all(v is None for v in (acc, prec, f1)):
            continue
        labels.append(label)
        accs.append(acc if acc is not None else 0.0)
        precs.append(prec if prec is not None else 0.0)
        f1s.append(f1 if f1 is not None else 0.0)

    if not labels:
        logger.warning(
            "test_metrics_comparison: no comparable metrics found for chart generation. metric_keys=%s",
            sorted((metrics or {}).keys()),
        )
        return

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4))
    r1 = ax.bar(x - width, accs, width, label="Accuracy", color="steelblue")
    r2 = ax.bar(x, precs, width, label="Precision (macro)", color="mediumseagreen")
    r3 = ax.bar(x + width, f1s, width, label="F1 (macro)", color="coral")

    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    # Annotate bars
    for rect in list(r1) + list(r2) + list(r3):
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    plt.savefig("test_metrics_comparison.png", dpi=150)
    plt.close(fig)

