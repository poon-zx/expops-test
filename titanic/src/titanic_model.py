from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from expops.core import SerializableData, log_metric, process, step

logger = logging.getLogger(__name__)


def _titanic_csv_path() -> Path:
    return Path("data/Titanic-Dataset.csv")


def _safe_median(series: pd.Series, default: float = 0.0) -> float:
    s = pd.to_numeric(series, errors="coerce")
    med = float(s.median()) if s.notnull().any() else float(default)
    if not np.isfinite(med):
        return float(default)
    return float(med)


def _coerce_binary_target(y: pd.Series) -> np.ndarray:
    yy = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    yy = np.clip(yy.to_numpy(), 0, 1)
    return yy.astype(int)


def _clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Embarked" in df.columns:
        embarked_mode = df["Embarked"].mode(dropna=True)
        fill = embarked_mode.iloc[0] if not embarked_mode.empty else "S"
        df["Embarked"] = df["Embarked"].fillna(fill)

    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        df["Fare"] = df["Fare"].fillna(_safe_median(df["Fare"], default=0.0))

    if "Name" in df.columns:
        df["Title"] = df["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        common = {"Mr", "Mrs", "Miss", "Master"}
        df["Title"] = df["Title"].fillna("Unknown").apply(lambda t: t if t in common else "Rare")
    else:
        df["Title"] = "Mr"

    sibsp = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0).astype(int) if "SibSp" in df.columns else 0
    parch = pd.to_numeric(df["Parch"], errors="coerce").fillna(0).astype(int) if "Parch" in df.columns else 0
    df["FamilySize"] = sibsp + parch + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"] = df.groupby("Title")["Age"].transform(lambda s: s.fillna(s.median()))
        df["Age"] = df["Age"].fillna(_safe_median(df["Age"], default=28.0))
    else:
        df["Age"] = 28.0

    keep_cols = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def _encode_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    required = {"Survived", "Sex", "Pclass"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for Titanic preprocessing: {sorted(missing)}")

    y = _coerce_binary_target(df["Survived"])

    cat_cols = [c for c in ["Sex", "Embarked", "Title", "Pclass"] if c in df.columns]
    num_cols = [c for c in ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"] if c in df.columns]

    X_df = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        X_df[c] = X_df[c].astype(str)
    for c in num_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(_safe_median(X_df[c], default=0.0)).astype(float)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    X_all = preprocessor.fit_transform(X_df)
    X_all = np.asarray(X_all, dtype=float)
    return X_all, y


@process()
def data_preprocessing(
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Load the Titanic dataset, engineer features, and create a train/test split.

    This process:
    - Loads `data/Titanic-Dataset.csv`.
    - Cleans and imputes missing values (e.g., `Age`, `Fare`, `Embarked`).
    - Engineers additional features (`Title`, `FamilySize`, `IsAlone`).
    - One-hot encodes categorical features, scales numeric features, and splits
      into train/test sets.

    Parameters
    ----------
    random_seed : int, default=42
        Seed for the train/test split random state.

    Returns
    -------
    dict
        Dictionary containing:

        - **X_train** (`list[list[float]]`): Training feature matrix.
        - **X_test** (`list[list[float]]`): Test feature matrix.
        - **y_train** (`list[int]`): Training labels in `{0, 1}`.
        - **y_test** (`list[int]`): Test labels in `{0, 1}`.

    Raises
    ------
    FileNotFoundError
        If the source CSV does not exist at the expected path.
    ValueError
        If required columns are missing or feature encoding fails.
    """
    test_size = 0.2
    _ =  None
    @step()
    def load_csv() -> SerializableData:
        """
        Load the Titanic CSV and return a serializable payload.

        Returns
        -------
        dict
            Dictionary containing:

            - **df** (`dict[str, list]`): Raw dataframe serialized as column-oriented lists.

        Raises
        ------
        FileNotFoundError
            If the source CSV does not exist at the expected path.
        """
        path = _titanic_csv_path()
        if not path.exists():
            raise FileNotFoundError(f"Titanic CSV not found at {path}")
        df = pd.read_csv(path)
        return {"df": df.to_dict(orient="list")}

    @step()
    def clean_and_engineer(raw: SerializableData) -> SerializableData:
        """
        Clean raw Titanic rows and engineer additional features.

        Parameters
        ----------
        raw : dict
            Serializable payload containing **df** as a column-oriented dictionary.

        Returns
        -------
        dict
            Dictionary containing:

            - **df** (`dict[str, list]`): Cleaned dataframe serialized as column-oriented lists.
        """
        df = pd.DataFrame(raw["df"])
        out = _clean_and_engineer(df)
        return {"df": out.to_dict(orient="list")}

    @step()
    def encode_and_split(raw: SerializableData, test_size: float = 0.2, random_seed: int = 42) -> SerializableData:
        """
        Encode engineered features and create a train/test split.

        Parameters
        ----------
        raw : dict
            Serializable payload containing **df** as a column-oriented dictionary.
        test_size : float, default=0.2
            Fraction of samples allocated to the test split.
        random_seed : int, default=42
            Seed for the train/test split random state.

        Returns
        -------
        dict
            Dictionary containing:

            - **X_train** (`list[list[float]]`): Training feature matrix.
            - **X_test** (`list[list[float]]`): Test feature matrix.
            - **y_train** (`list[int]`): Training labels in `{0, 1}`.
            - **y_test** (`list[int]`): Test labels in `{0, 1}`.
            - **n_train** (`int`): Number of training samples.
            - **n_test** (`int`): Number of test samples.

        Raises
        ------
        ValueError
            If required columns are missing or encoding fails.
        """
        df = pd.DataFrame(raw["df"])
        X_all, y = _encode_features(df)

        rs = int(random_seed) if random_seed is not None else None
        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y,
            test_size=float(test_size),
            shuffle=True,
            stratify=stratify,
            random_state=rs,
        )

        return {
            "X_train": X_train.astype(float).tolist(),
            "X_test": X_test.astype(float).tolist(),
            "y_train": y_train.astype(int).tolist(),
            "y_test": y_test.astype(int).tolist(),
        }

    raw = load_csv()
    engineered = clean_and_engineer(raw=raw)
    return encode_and_split(raw=engineered, test_size=test_size, random_seed=random_seed)


@step()
def train_and_log_nn(
    X_train: SerializableData,
    y_train: SerializableData,
    hidden_layers: list[int] | tuple[int, ...] | None = None,
    learning_rate: float = 0.01,
    epochs: int = 50,
    batch_size: int = 64,
    random_seed: int | None = None,
    branch_name: str = "",
) -> Dict[str, Any]:
    """
    Train an MLP classifier and log training loss over epochs.

    Training is performed by repeatedly calling `.fit(...)` with
    `warm_start=True` and `max_iter=1` to emulate epochs and enable per-epoch
    metric logging.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix with shape `(n_train, n_features)`.
    y_train : array-like
        Training labels with shape `(n_train,)` and values in `{0, 1}`.
    hidden_layers : list[int] or tuple[int, ...] or None, default=None
        Hidden layer sizes passed to `MLPClassifier(hidden_layer_sizes=...)`.
        Defaults to `(64, 32)` when not provided.
    learning_rate : float, default=0.01
        Initial learning rate for the MLP optimizer.
    epochs : int, default=50
        Number of training epochs to simulate.
    batch_size : int, default=64
        Batch size used by the classifier (clipped to `n_train`).
    random_seed : int or None, default=None
        Random seed for the classifier.
    branch_name : str, default=""
        Optional name used to label logged warnings.

    Returns
    -------
    dict
        Dictionary containing:

        - **model** (`sklearn.neural_network.MLPClassifier`): Trained classifier.

    Raises
    ------
    ValueError
        If `X_train` or `y_train` are empty.
    """
    X = np.asarray(X_train or [], dtype=float)
    y = np.asarray(y_train or [], dtype=int)
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty training data provided to NN training step")

    hidden_layers = tuple(hidden_layers or [64, 32])
    learning_rate = float(learning_rate)
    epochs = int(epochs)
    batch_size = int(batch_size)
    rs = int(random_seed) if random_seed is not None else None

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        solver="adam",
        activation="relu",
        alpha=0.001,
        batch_size=min(batch_size, X.shape[0]),
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=rs,
        verbose=False,
    )

    for epoch in range(epochs):
        clf.fit(X, y)
        try:
            if hasattr(clf, "loss_"):
                log_metric("train_loss", float(clf.loss_), step=epoch + 1)
        except Exception as e:
            logger.warning(f"[{branch_name or 'nn'}] Failed to log train_loss @epoch {epoch + 1}: {e}")

    return {"model": clf}


@process()
def nn_training_a(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layers: list[int] | tuple[int, ...] | None = None,
    learning_rate: float = 0.01,
    epochs: int = 50,
    batch_size: int = 64,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    """
    Train NN candidate A and attach the provided test split for inference.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix with shape `(n_train, n_features)`.
    y_train : array-like
        Training labels with shape `(n_train,)` and values in `{0, 1}`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        Test labels with shape `(n_test,)` and values in `{0, 1}`.
    hidden_layers : list[int] or tuple[int, ...] or None, default=None
        Hidden layer sizes passed to `MLPClassifier(hidden_layer_sizes=...)`.
    learning_rate : float, default=0.01
        Initial learning rate for the MLP optimizer.
    epochs : int, default=50
        Number of training epochs to simulate.
    batch_size : int, default=64
        Batch size used by the classifier (clipped to `n_train`).
    random_seed : int or None, default=None
        Random seed for the classifier.

    Returns
    -------
    dict
        Dictionary containing:

        - **model** (`sklearn.neural_network.MLPClassifier`): Trained classifier.
        - **X_test**: The provided `X_test`.
        - **y_test**: The provided `y_test`.
    """
    result = train_and_log_nn(
        X_train=X_train,
        y_train=y_train,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        branch_name="nn_training_a",
    )
    result["X_test"] = X_test
    result["y_test"] = y_test
    return result


@process()
def nn_training_b(
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layers: list[int] | tuple[int, ...] | None = None,
    learning_rate: float = 0.01,
    epochs: int = 50,
    batch_size: int = 64,
    random_seed: int | None = None,
) -> Dict[str, Any]:
    """
    Train NN candidate B and attach the provided test split for inference.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix with shape `(n_train, n_features)`.
    y_train : array-like
        Training labels with shape `(n_train,)` and values in `{0, 1}`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        Test labels with shape `(n_test,)` and values in `{0, 1}`.
    hidden_layers : list[int] or tuple[int, ...] or None, default=None
        Hidden layer sizes passed to `MLPClassifier(hidden_layer_sizes=...)`.
    learning_rate : float, default=0.01
        Initial learning rate for the MLP optimizer.
    epochs : int, default=50
        Number of training epochs to simulate.
    batch_size : int, default=64
        Batch size used by the classifier (clipped to `n_train`).
    random_seed : int or None, default=None
        Random seed for the classifier.

    Returns
    -------
    dict
        Dictionary containing:

        - **model** (`sklearn.neural_network.MLPClassifier`): Trained classifier.
        - **X_test**: The provided `X_test`.
        - **y_test**: The provided `y_test`.
    """
    result = train_and_log_nn(
        X_train=X_train,
        y_train=y_train,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        branch_name="nn_training_b",
    )
    result["X_test"] = X_test
    result["y_test"] = y_test
    return result


@process()
def linear_training(
    X_train,
    y_train,
    X_test,
    y_test,
    C: float = 0.9,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 200,
) -> Dict[str, Any]:
    """
    Train a logistic regression classifier and attach the test split.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix with shape `(n_train, n_features)`.
    y_train : array-like
        Training labels with shape `(n_train,)` and values in `{0, 1}`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        Test labels with shape `(n_test,)` and values in `{0, 1}`.
    C : float, default=0.9
        Inverse regularization strength.
    penalty : str, default="l2"
        Norm used in the penalization.
    solver : str, default="lbfgs"
        Optimization algorithm used by `LogisticRegression`.
    max_iter : int, default=200
        Maximum number of iterations for the solver.

    Returns
    -------
    dict
        Dictionary containing:

        - **model** (`sklearn.linear_model.LogisticRegression`): Trained classifier.
        - **X_test**: The provided `X_test`.
        - **y_test**: The provided `y_test`.

    Raises
    ------
    ValueError
        If `X_train` or `y_train` are empty.
    """
    X = np.asarray(X_train or [], dtype=float)
    y = np.asarray(y_train or [], dtype=int)
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty training data provided to Linear model training process")

    model = LogisticRegression(C=float(C), penalty=str(penalty), solver=str(solver), max_iter=int(max_iter))
    model.fit(X, y)
    return {"model": model, "X_test": X_test, "y_test": y_test}


@step()
def test_inference_binary(model: SerializableData, X_test: SerializableData, y_test: SerializableData) -> Dict[str, Any]:
    """
    Run binary classification inference and compute common metrics.

    Parameters
    ----------
    model : sklearn.base.ClassifierMixin
        Trained estimator implementing `predict(X)`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        True labels with shape `(n_test,)` and values in `{0, 1}`.

    Returns
    -------
    dict
        Dictionary containing:

        - **test_accuracy** (`float`): Accuracy score.
        - **test_precision** (`float`): Precision score (positive class).
        - **test_f1** (`float`): F1 score (positive class).

    Raises
    ------
    ValueError
        If `model` is missing when non-empty test data is provided.
    """
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

    if model is None:
        raise ValueError("Missing upstream model for inference")

    y_pred = np.asarray(model.predict(X), dtype=int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        log_metric("test_accuracy", acc)
        log_metric("test_precision", prec)
        log_metric("test_f1", f1)
    except Exception:
        pass

    return {"test_accuracy": acc, "test_precision": prec, "test_f1": f1}


@process()
def linear_inference(model, X_test, y_test) -> Dict[str, Any]:
    """
    Run inference for the linear model and emit metrics payload.

    Parameters
    ----------
    model : sklearn.base.ClassifierMixin
        Trained estimator implementing `predict(X)`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        True labels with shape `(n_test,)` and values in `{0, 1}`.

    Returns
    -------
    dict
        - **linear_inference** (`dict`): Inference metrics payload.
    """
    result = test_inference_binary(model=model, X_test=X_test, y_test=y_test)
    return {"linear_inference": result}


@process()
def nn_inference_a(model, X_test, y_test) -> Dict[str, Any]:
    """
    Run inference for NN candidate A and emit metrics payload.

    Parameters
    ----------
    model : sklearn.base.ClassifierMixin
        Trained estimator implementing `predict(X)`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        True labels with shape `(n_test,)` and values in `{0, 1}`.

    Returns
    -------
    dict
        - **nn_inference_a** (`dict`): Inference metrics payload.
    """
    result = test_inference_binary(model=model, X_test=X_test, y_test=y_test)
    return {"nn_inference_a": result}


@process()
def nn_inference_b(model, X_test, y_test) -> Dict[str, Any]:
    """
    Run inference for NN candidate B and emit metrics payload.

    Parameters
    ----------
    model : sklearn.base.ClassifierMixin
        Trained estimator implementing `predict(X)`.
    X_test : array-like
        Test feature matrix with shape `(n_test, n_features)`.
    y_test : array-like
        True labels with shape `(n_test,)` and values in `{0, 1}`.

    Returns
    -------
    dict
        - **nn_inference_b** (`dict`): Inference metrics payload.
    """
    result = test_inference_binary(model=model, X_test=X_test, y_test=y_test)
    return {"nn_inference_b": result}

@process()
def partition_aggregate(linear_inference: Dict[str, Any], nn_inference_a: Dict[str, Any], nn_inference_b: Dict[str, Any]):
    """
    Aggregate results from data-parallel branches (currently a no-op).

    Parameters
    ----------
    rows : dict[str, Any] or None, default=None
        Placeholder for aggregated branch outputs.
    """
    return {}