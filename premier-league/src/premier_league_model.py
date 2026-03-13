from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

from expops.core import (
    step,
    process,
    SerializableData,
    log_metric,
)

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Resolve the current project's root directory based on workspace + project env."""
    try:
        ws_raw = os.environ.get("MLOPS_WORKSPACE_DIR") or "."
        ws = Path(ws_raw).expanduser().resolve()
    except Exception:
        ws = Path.cwd()
    pid = os.environ.get("MLOPS_PROJECT_ID")
    if pid:
        return ws / pid
    return ws


def _csv_path() -> Path:
    return _project_root() / "data" / "England CSV.csv"


def _get_result_column_name(df: pd.DataFrame) -> str:
    if "FT Result" in df.columns:
        return "FT Result"
    if "FTR" in df.columns:
        return "FTR"
    raise ValueError("Missing required result column: expected 'FT Result' or 'FTR'")


def _derive_outcome_labels(df: pd.DataFrame) -> np.ndarray:
    result_col = _get_result_column_name(df)
    mapping = {"H": 0, "D": 1, "A": 2}
    y = df[result_col].astype(str).map(mapping)
    if y.isnull().any():
        bad = df.loc[y.isnull(), result_col].unique().tolist()
        raise ValueError(f"Unexpected values in {result_col}: {bad}")
    return y.astype(int).to_numpy()


def _get_cat_num_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [c for c in ["Season", "HomeTeam", "AwayTeam", "Referee", "League"] if c in df.columns]
    num_cols = [
        c
        for c in [
            "HTH Goals",
            "HTA Goals",
            "H Shots",
            "A Shots",
            "H SOT",
            "A SOT",
            "H Fouls",
            "A Fouls",
            "H Corners",
            "A Corners",
            "H Yellow",
            "A Yellow",
            "H Red",
            "A Red",
            "Display_Order",
            "DayOfWeek",
            "Month",
        ]
        if c in df.columns
    ]
    return cat_cols, num_cols


def _derive_inference_key(train_key: str, fallback: str) -> str:
    if isinstance(train_key, str) and "training" in train_key:
        return train_key.replace("training", "inference")
    return fallback


def _build_features_dataframe(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    X_df = pd.DataFrame(index=df.index)
    # Numeric
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isnull().any():
            med = s.median()
            s = s.fillna(med if not np.isnan(med) else 0)
        X_df[c] = s.astype(float)
    for c in cat_cols:
        X_df[c] = df[c].astype(str)
    for drop_c in ["FT Result", "FTR", "HT Result", "Date"]:
        if drop_c in X_df.columns:
            X_df = X_df.drop(columns=[drop_c])
    return X_df


@process()
def feature_engineering_generic(
    test_size: float = 0.2,
    pca_components: int = 16,
    random_seed: int = 42,
):
    """Load CSV, parse dates, derive labels (H/D/A), stratified split indices, and log analysis metrics."""

    @step()
    def load_csv():
        path = _csv_path()
        if not path.exists():
            raise FileNotFoundError(f"Premier League CSV not found at {path}")
        df = pd.read_csv(path)
        try:
            logger.info(f"[feature_engineering_generic.load_csv] Loaded df shape: {df.shape}")
        except Exception:
            pass
        return {"df": df.to_dict(orient="list")}

    @step()
    def derive_labels_and_indices(raw: SerializableData, test_size: float = 0.2):
        df = pd.DataFrame(raw["df"])
        # Parse date-based features
        if "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["DayOfWeek"] = dt.dt.weekday.fillna(0).astype(int)
            df["Month"] = dt.dt.month.fillna(1).astype(int)
        else:
            df["DayOfWeek"] = 0
            df["Month"] = 1

        y = _derive_outcome_labels(df)

        # Goals histograms for static charts
        hist_home = {}
        hist_away = {}
        if "FTH Goals" in df.columns and "FTA Goals" in df.columns:
            goals_home = pd.to_numeric(df["FTH Goals"], errors="coerce").fillna(0).astype(int)
            goals_away = pd.to_numeric(df["FTA Goals"], errors="coerce").fillna(0).astype(int)
            hist_home = goals_home.value_counts().sort_index().astype(int).to_dict()
            hist_away = goals_away.value_counts().sort_index().astype(int).to_dict()
            log_metric("goals_hist_home", hist_home)
            log_metric("goals_hist_away", hist_away)

        return {
            "df": df.to_dict(orient="list"),
            "columns": [str(c) for c in df.columns],
            "labels": y.astype(int).tolist(),
        }

    @step()
    def feature_analysis(
        basic: SerializableData,
        pca_components: int = 16,
        random_seed: int = 42,
    ):
        df = pd.DataFrame(basic["df"])
        if "DayOfWeek" not in df.columns or "Month" not in df.columns:
            if "Date" in df.columns:
                dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                df["DayOfWeek"] = dt.dt.weekday.fillna(0).astype(int)
                df["Month"] = dt.dt.month.fillna(1).astype(int)
            else:
                df["DayOfWeek"] = 0
                df["Month"] = 1

        cat_cols, num_cols = _get_cat_num_cols(df)
        X_df = _build_features_dataframe(df, cat_cols, num_cols)

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", encoder, cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )

        X_all = preprocessor.fit_transform(X_df)
        n_components = min(int(pca_components), X_all.shape[1]) if X_all.shape[1] > 0 else 0
        if n_components > 0:
            pca = PCA(n_components=n_components, random_state=int(random_seed))
            _ = pca.fit_transform(X_all)
            evr = pca.explained_variance_ratio_.tolist()
            cum = np.cumsum(pca.explained_variance_ratio_).tolist()
        else:
            evr = []
            cum = []

        log_metric("pca_explained_variance_ratio", evr)
        log_metric("pca_cumulative_variance", cum)
        return {}

    raw = load_csv()
    basic = derive_labels_and_indices(raw=raw, test_size=test_size)
    _ = feature_analysis(basic=basic, pca_components=pca_components, random_seed=random_seed)
    return basic


@process()
def nn_data_parallel(
    df,
):
    """Prepare row-wise data for data parallelism split."""
    frame = pd.DataFrame(df)
    rows = frame.to_dict(orient="records")
    return rows


@process()
def nn_partition_aggregate(rows: Dict[str, Any] | None = None):
    """No-op aggregation placeholder for data-parallel branches."""
    return {}


@process()
def preprocess_linear_nn(
    df,
    test_size: float = 0.2,
    random_seed: int = 42,
    columns: list[str] | None = None,
):
    """Preprocess for Linear/NN: OHE categorical + StandardScaler numeric."""
    df = pd.DataFrame(df)
    y = _derive_outcome_labels(df)
    idx = np.arange(len(df))
    rs = int(random_seed) if random_seed is not None else None
    try:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=float(test_size),
            shuffle=True,
            stratify=y,
            random_state=rs,
        )
    except Exception:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=float(test_size),
            shuffle=True,
            random_state=rs,
        )

    # Date-derived columns already present from FE; if not, add defaults
    if "DayOfWeek" not in df.columns or "Month" not in df.columns:
        if "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["DayOfWeek"] = dt.dt.weekday.fillna(0).astype(int)
            df["Month"] = dt.dt.month.fillna(1).astype(int)
        else:
            df["DayOfWeek"] = 0
            df["Month"] = 1

    cat_cols, num_cols = _get_cat_num_cols(df)
    X_df = _build_features_dataframe(df, cat_cols, num_cols)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_df.iloc[idx_train])
    X_test = preprocessor.transform(X_df.iloc[idx_test])
    y_train = y[idx_train]
    y_test = y[idx_test]

    return {
        "X_train": X_train.astype(float).tolist(),
        "X_test": X_test.astype(float).tolist(),
        "y_train": y_train.astype(int).tolist(),
        "y_test": y_test.astype(int).tolist(),
        "row_indices_train": idx_train.astype(int).tolist(),
        "row_indices_test": idx_test.astype(int).tolist(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


@process()
def preprocess_xgb(
    df,
    test_size: float = 0.2,
    random_seed: int = 42,
    columns: list[str] | None = None,
):
    """Preprocess for XGB: OHE categorical only (no scaling)."""
    df = pd.DataFrame(df)
    y = _derive_outcome_labels(df)
    idx = np.arange(len(df))
    rs = int(random_seed) if random_seed is not None else None
    try:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=float(test_size),
            shuffle=True,
            stratify=y,
            random_state=rs,
        )
    except Exception:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=float(test_size),
            shuffle=True,
            random_state=rs,
        )

    if "DayOfWeek" not in df.columns or "Month" not in df.columns:
        if "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["DayOfWeek"] = dt.dt.weekday.fillna(0).astype(int)
            df["Month"] = dt.dt.month.fillna(1).astype(int)
        else:
            df["DayOfWeek"] = 1
            df["Month"] = 1

    cat_cols, num_cols = _get_cat_num_cols(df)
    X_df = _build_features_dataframe(df, cat_cols, num_cols)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_df.iloc[idx_train])
    X_test = preprocessor.transform(X_df.iloc[idx_test])
    y_train = y[idx_train]
    y_test = y[idx_test]

    return {
        "X_train": X_train.astype(float).tolist(),
        "X_test": X_test.astype(float).tolist(),
        "y_train": y_train.astype(int).tolist(),
        "y_test": y_test.astype(int).tolist(),
        "row_indices_train": idx_train.astype(int).tolist(),
        "row_indices_test": idx_test.astype(int).tolist(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


@step()
def train_logistic_classifier(prep_data: SerializableData, logreg_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    X_train = np.asarray(prep_data.get("X_train", []), dtype=float)
    y_train = np.asarray(prep_data.get("y_train", []), dtype=int)
    if X_train.size == 0:
        raise ValueError("Empty training data provided to Logistic training step")

    params = logreg_params or {}
    max_iter = int(params.get("max_iter", 500))
    class_weight = params.get("class_weight", None)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return {"model": model}


@step()
def train_and_evaluate_nn_classifier(
    prep_data: SerializableData,
    hidden_layers: list[int] | tuple[int, ...] | None = None,
    learning_rate: float = 0.001,
    epochs: int = 50,
    random_seed: int | None = None,
    branch_name: str = "",
) -> Dict[str, Any]:
    hidden_layers = tuple(hidden_layers or [128, 64])
    learning_rate = float(learning_rate)
    epochs = int(epochs)
    if random_seed is None:
        random_seed = 30
    random_seed = int(random_seed)

    X_train = np.asarray(prep_data.get("X_train", []), dtype=float)
    y_train = np.asarray(prep_data.get("y_train", []), dtype=int)
    if X_train.size == 0:
        raise ValueError("Empty training data provided to NN classifier training step")

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        shuffle=True,
        random_state=random_seed,
        verbose=False,
    )

    for epoch in range(epochs):
        clf.fit(X_train, y_train)
        try:
            if hasattr(clf, "loss_"):
                log_metric("train_loss", float(clf.loss_), step=epoch + 1)
            preds = clf.predict(X_train)
            f1 = float(f1_score(y_train, preds, average="macro"))
            log_metric("train_f1", f1, step=epoch + 1)
        except Exception as e:
            logger.warning(f"[{branch_name or 'nn'}] Failed to log training metrics @epoch {epoch + 1}: {e}")
    return {"model": clf}


@step()
def train_xgb_classifier(prep_data: SerializableData, xgb_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    xgb_params = xgb_params or {}
    params = {
        "n_estimators": int(xgb_params.get("n_estimators", 400)),
        "max_depth": int(xgb_params.get("max_depth", 4)),
        "learning_rate": float(xgb_params.get("learning_rate", 0.1)),
        "subsample": float(xgb_params.get("subsample", 0.9)),
        "colsample_bytree": float(xgb_params.get("colsample_bytree", 0.9)),
        "n_jobs": int(xgb_params.get("n_jobs", 1)),
        "verbosity": 0,
        "random_state": int(xgb_params.get("random_state", 42)) if "random_state" in xgb_params else None,
        "tree_method": xgb_params.get("tree_method", "auto"),
        "objective": "multi:softprob",
        "num_class": 3,
    }

    params = {k: v for k, v in params.items() if v is not None}

    X_train = np.asarray(prep_data.get("X_train", []), dtype=float)
    y_train = np.asarray(prep_data.get("y_train", []), dtype=int)
    if X_train.size == 0:
        raise ValueError("Empty training data provided to XGB classifier training step")

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return {"model": model}


@step()
def test_inference_classification(model: SerializableData, X_test: SerializableData, y_test: SerializableData) -> Dict[str, Any]:
    X = np.asarray(X_test or [], dtype=float)
    y_true = np.asarray(y_test or [], dtype=int)
    if X.size == 0 or y_true.size == 0:
        try:
            log_metric("test_accuracy", 0.0)
            log_metric("test_precision", 0.0)
            log_metric("test_f1", 0.0)
        except Exception:
            pass
        return {"test_accuracy": 0.0, "test_precision": 0.0, "test_f1": 0.0}

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):
            proba = np.stack(proba, axis=-1)
        if proba.ndim == 3:
            proba = proba
    else:
        preds = model.predict(X)
        n_classes = len(np.unique(y_true))
        proba = np.eye(n_classes)[preds]

    y_pred = np.asarray(np.argmax(proba, axis=1), dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    try:
        log_metric("test_accuracy", acc)
        log_metric("test_precision", prec)
        log_metric("test_f1", f1)
    except Exception:
        pass

    return {"test_accuracy": acc, "test_precision": prec, "test_f1": f1}


@process()
def linear_training(
    X_train,
    y_train,
    X_test,
    y_test,
    row_indices_test,
    logreg_params: Dict[str, Any] | None = None,
):
    prep = {"X_train": X_train, "y_train": y_train}
    result = train_logistic_classifier(prep_data=prep, logreg_params=logreg_params)
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["row_indices_test"] = row_indices_test
    return result


@process()
def define_nn_training_process(
    X_train,
    y_train,
    X_test,
    y_test,
    row_indices_test,
    hidden_layers: list[int] | tuple[int, ...] | None = None,
    learning_rate: float = 0.001,
    epochs: int = 50,
    random_seed: int | None = None,
    branch_name: str = "",
):
    prep = {"X_train": X_train, "y_train": y_train}
    result = train_and_evaluate_nn_classifier(
        prep_data=prep,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        random_seed=random_seed,
        branch_name=branch_name,
    )
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["row_indices_test"] = row_indices_test
    return result


@process()
def define_xgb_training_process(
    X_train,
    y_train,
    X_test,
    y_test,
    row_indices_test,
    **xgb_params: Any,
):
    prep = {"X_train": X_train, "y_train": y_train}
    result = train_xgb_classifier(prep_data=prep, xgb_params=xgb_params or None)
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["row_indices_test"] = row_indices_test
    return result


@process()
def linear_inference(model, X_test, y_test, row_indices_test):
    result = test_inference_classification(model=model, X_test=X_test, y_test=y_test)
    result["model"] = model
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["row_indices_test"] = row_indices_test
    result["source_training"] = "linear_training"
    return {"linear_inference": result}


@process()
def define_nn_inference_process(
    model,
    X_test,
    y_test,
    row_indices_test,
    train_key: str = "nn_training_a",
):
    result = test_inference_classification(model=model, X_test=X_test, y_test=y_test)
    result["model"] = model
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["row_indices_test"] = row_indices_test
    result["source_training"] = str(train_key)
    inference_key = _derive_inference_key(str(train_key), "nn_inference")
    return {inference_key: result}


@process()
def define_xgb_inference_process(
    model,
    X_test,
    y_test,
    row_indices_test,
    train_key: str = "xgb_training_a",
):
    result = test_inference_classification(model=model, X_test=X_test, y_test=y_test)
    result["model"] = model
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["row_indices_test"] = row_indices_test
    result["source_training"] = str(train_key)
    inference_key = _derive_inference_key(str(train_key), "xgb_inference")
    return {inference_key: result}


@process()
def nn_best_selection(nn_inference_a, nn_inference_b):
    inf_a = nn_inference_a or {}
    inf_b = nn_inference_b or {}
    f1_a = float(inf_a.get("test_f1", 0.0) or 0.0)
    f1_b = float(inf_b.get("test_f1", 0.0) or 0.0)

    best_key = "nn_training_a"
    best_f1 = f1_a
    best_inf = inf_a
    if f1_b >= f1_a:
        best_key = "nn_training_b"
        best_f1 = f1_b
        best_inf = inf_b

    return {
        "nn_best_selection": {
            "model": best_inf.get("model"),
            "X_test": best_inf.get("X_test"),
            "y_test": best_inf.get("y_test"),
            "row_indices_test": best_inf.get("row_indices_test"),
            "f1": best_f1,
            "best_key": best_key,
        }
    }


@process()
def xgb_best_selection(xgb_inference_a, xgb_inference_b):
    inf_a = xgb_inference_a or {}
    inf_b = xgb_inference_b or {}
    f1_a = float(inf_a.get("test_f1", 0.0) or 0.0)
    f1_b = float(inf_b.get("test_f1", 0.0) or 0.0)

    best_key = "xgb_training_a"
    best_f1 = f1_a
    best_inf = inf_a
    if f1_b >= f1_a:
        best_key = "xgb_training_b"
        best_f1 = f1_b
        best_inf = inf_b

    return {
        "xgb_best_selection": {
            "model": best_inf.get("model"),
            "X_test": best_inf.get("X_test"),
            "y_test": best_inf.get("y_test"),
            "row_indices_test": best_inf.get("row_indices_test"),
            "f1": best_f1,
            "best_key": best_key,
        }
    }


@process()
def nn_best_inference(nn_best_selection):
    sel = nn_best_selection or {}
    result = test_inference_classification(model=sel.get("model"), X_test=sel.get("X_test"), y_test=sel.get("y_test"))
    return {"nn_best_inference": result}


@process()
def xgb_best_inference(xgb_best_selection):
    sel = xgb_best_selection or {}
    result = test_inference_classification(model=sel.get("model"), X_test=sel.get("X_test"), y_test=sel.get("y_test"))
    return {"xgb_best_inference": result}


@process()
def ensemble_inference(
    model,
    X_test,
    y_test,
    row_indices_test,
    nn_best_selection,
    xgb_best_selection,
):
    xgb_sel = xgb_best_selection or {}

    lin_model = model
    xgb_model = xgb_sel.get("model")

    X_lin = np.asarray(X_test or [], dtype=float)
    y_true = np.asarray(y_test or [], dtype=int)
    idx_lin = np.asarray(row_indices_test or [], dtype=int)

    X_xgb = np.asarray(xgb_sel.get("X_test") or [], dtype=float)
    idx_xgb = np.asarray(xgb_sel.get("row_indices_test") or [], dtype=int)

    # Default to equal weighting when no prior metrics are provided.
    w_lin = 1.0
    w_xgb = 1.0

    weights = np.array([w_lin, w_xgb], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.array([1.0, 1.0], dtype=float)
    weights = weights / weights.sum()

    # Predict probabilities
    def _predict_proba_safe(m, X):
        if m is None or X.size == 0:
            return None
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)
            if isinstance(p, list):
                p = np.stack(p, axis=-1)
            return p
        preds = m.predict(X)
        n_classes = 3
        return np.eye(n_classes)[preds]

    P_lin = _predict_proba_safe(lin_model, X_lin)
    P_xgb = _predict_proba_safe(xgb_model, X_xgb)

    # Align by row indices if provided
    def _align_to(reference_idx, idx_other, P_other):
        if P_other is None or reference_idx.size == 0 or idx_other.size == 0:
            return None
        if np.array_equal(reference_idx, idx_other):
            return P_other
        order = {int(v): i for i, v in enumerate(idx_other.tolist())}
        aligned = np.zeros_like(P_other)
        for pos, rid in enumerate(reference_idx.tolist()):
            j = order.get(int(rid))
            if j is None:
                continue
            aligned[pos] = P_other[j]
        return aligned

    P_xgb_aligned = _align_to(idx_lin, idx_xgb, P_xgb) if P_xgb is not None else None

    # Combine probabilities (weighted soft vote)
    probas = []
    wlist = []
    if P_lin is not None:
        probas.append(P_lin)
        wlist.append(weights[0])
    if P_xgb_aligned is not None:
        probas.append(P_xgb_aligned)
        wlist.append(weights[1])

    if not probas or y_true.size == 0:
        try:
            log_metric("test_accuracy", 0.0)
            log_metric("test_precision", 0.0)
            log_metric("test_f1", 0.0)
        except Exception:
            pass
        return {"ensemble_inference": {"test_accuracy": 0.0, "test_precision": 0.0, "test_f1": 0.0}}

    W = np.array(wlist, dtype=float)
    W = W / W.sum()
    stacked = np.stack(probas, axis=0)
    ens = np.tensordot(W, stacked, axes=(0, 0))
    y_pred = np.argmax(ens, axis=1).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    try:
        log_metric("test_accuracy", acc)
        log_metric("test_precision", prec)
        log_metric("test_f1", f1)
    except Exception:
        pass
    return {"ensemble_inference": {"test_accuracy": acc, "test_precision": prec, "test_f1": f1}}

