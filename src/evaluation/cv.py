"""Validación cruzada estratificada: misma partición para todos los experimentos."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 985
N_SPLITS = 5


def get_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold | None = None) -> tuple[float, float]:
    """Entrena `model` en cada fold de `cv` y mide AUC-ROC en el fold de validación.

    Devuelve (auc_media, auc_std) sobre los folds. `model` debe soportar
    `fit`/`predict_proba` (p. ej. LGBMClassifier, LogisticRegression, o un Pipeline).
    """
    cv = cv if cv is not None else get_cv()
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        est = clone(model)
        est.fit(X_train, y_train)
        proba = est.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, proba))
    scores = np.array(scores)
    return scores.mean(), scores.std()
