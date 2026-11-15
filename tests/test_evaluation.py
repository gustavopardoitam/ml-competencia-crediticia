import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.cv import evaluate, get_cv
from evaluation.log import log_experiment, read_log


def _make_separable_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x = y + rng.normal(0, 0.1, size=n)  # variable casi perfectamente separable
    return pd.DataFrame({"x": x}), pd.Series(y)


def test_get_cv_splits_son_estratificados():
    X, y = _make_separable_data(n=100)
    folds = list(get_cv(n_splits=5).split(X, y))
    assert len(folds) == 5
    for _, val_idx in folds:
        assert abs(y.iloc[val_idx].mean() - y.mean()) < 0.15


def test_evaluate_detecta_senal_fuerte():
    X, y = _make_separable_data()
    auc_mean, auc_std = evaluate(LogisticRegression(), X, y)
    assert auc_mean > 0.95
    assert auc_std >= 0


def test_evaluate_es_reproducible():
    X, y = _make_separable_data()
    assert evaluate(LogisticRegression(), X, y) == evaluate(LogisticRegression(), X, y)


def test_log_experiment_agrega_filas(tmp_path):
    log_path = tmp_path / "experiments.csv"
    log_experiment("baseline", "crudas", "LogReg", 0.86, 0.01, log_path=log_path)
    log_experiment(
        "con_features", "prepare()", "LightGBM", 0.88, 0.009,
        kaggle_score=0.865, log_path=log_path,
    )
    df = read_log(log_path)
    assert list(df["nombre"]) == ["baseline", "con_features"]
    assert pd.isna(df["kaggle_score"].iloc[0])
    assert df["kaggle_score"].iloc[1] == 0.865
