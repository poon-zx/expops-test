from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from expops.core import log_metric, process


def _project_root() -> Path:
    """
    Resolve the current project's root directory.

    Uses MLOPS_WORKSPACE_DIR and MLOPS_PROJECT_ID when available so that
    data lives under <workspace>/<project_id>/data.
    """
    try:
        ws_raw = os.environ.get("MLOPS_WORKSPACE_DIR") or "."
        ws = Path(ws_raw).expanduser().resolve()
    except Exception:
        ws = Path.cwd()
    pid = os.environ.get("MLOPS_PROJECT_ID")
    if pid:
        return ws / pid
    return ws


def _load_xy(csv_path: str | Path):
    df = pd.read_csv(Path(csv_path))
    y = df.pop("label").values
    x = df.values
    return x, y


def _training_csv_path() -> Path:
    return _project_root() / "data" / "train.csv"


def _validation_csv_path() -> Path:
    return _training_csv_path()


@process()
def train_model():
    """
    Train a logistic regression classifier on the project training CSV.

    The training dataset is loaded from `<project_root>/data/train.csv` where
    `<project_root>` is resolved via `MLOPS_WORKSPACE_DIR` and `MLOPS_PROJECT_ID`
    when available.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Dictionary containing:

        - **model** (`sklearn.linear_model.LogisticRegression`): Fitted classifier.
    """
    train_path = _training_csv_path()
    x, y = _load_xy(train_path)
    model = LogisticRegression(max_iter=200)
    model.fit(x, y)

    train_acc = float(accuracy_score(y, model.predict(x)))
    log_metric("accuracy", train_acc)

    return {"model": model}


@process()
def evaluate_model(model):
    """
    Evaluate a trained classifier on the validation CSV and log accuracy.

    The validation dataset is currently the same path as the training dataset
    (`<project_root>/data/train.csv`).

    Parameters
    ----------
    model : sklearn.base.ClassifierMixin
        Trained estimator implementing `predict(X)`.

    Returns
    -------
    dict
        Empty dictionary. Metrics are logged via `log_metric`.

    Raises
    ------
    ValueError
        If `model` is None (missing upstream output from `train_model`).
    FileNotFoundError
        If the validation CSV does not exist at the expected path.
    """
    if model is None:
        raise ValueError("Missing upstream model. Expected `model` from training process.")

    val_path = _validation_csv_path()
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation CSV not found at {val_path}. "
            "Expected the template dataset at <project_root>/data/train.csv."
        )
    x, y = _load_xy(val_path)
    eval_acc = float(accuracy_score(y, model.predict(x)))
    log_metric("accuracy", eval_acc)

    return {}

