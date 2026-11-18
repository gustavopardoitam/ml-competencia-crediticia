"""Correr, registrar y comparar experimentos con la partición estándar de CV."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from evaluation.cv import evaluate_folds
from evaluation.log import LOG_PATH, log_experiment


def model_name(model) -> str:
    """Nombre corto del modelo: pasos de un Pipeline, o la clase del estimador."""
    if hasattr(model, "steps"):
        return " + ".join(paso for paso, _ in model.steps)
    return model.__class__.__name__


def run_experiment(
    nombre: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    features_desc: str,
    notas: str = "",
    cv: StratifiedKFold | None = None,
    log_path: Path = LOG_PATH,
) -> np.ndarray:
    """Evalúa `model` con la CV estándar, imprime y registra el resultado.

    Devuelve el AUC de cada fold, para compararlo de forma pareada con otros experimentos.
    """
    folds = evaluate_folds(model, X, y, cv)
    print(f"{nombre}: AUC {folds.mean():.5f} +/- {folds.std():.5f}")
    log_experiment(nombre=nombre, features=features_desc, modelo=model_name(model),
                   cv_auc_mean=folds.mean(), cv_auc_std=folds.std(), notas=notas,
                   log_path=log_path)
    return folds


def paired_comparison(candidates: dict[str, np.ndarray], reference: np.ndarray) -> pd.DataFrame:
    """Compara fold por fold cada candidato contra `reference` (misma partición de CV).

    La diferencia pareada elimina la variación que viene de qué tan difícil es cada fold,
    así que es más sensible que comparar las medias contra la std entre folds.
    """
    diff = pd.DataFrame({k: v - reference for k, v in candidates.items()})
    return pd.DataFrame({
        "mejora_media": diff.mean(),
        "std_pareada": diff.std(ddof=0),
        "folds_que_mejoran": (diff > 0).sum(),
    })
