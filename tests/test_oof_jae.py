"""Pruebas de la validación cruzada con predicciones OOF (src/evaluation/oof_jae.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluation.log import read_log
from evaluation.oof_jae import (
    combinar_resultados,
    evaluar_modelo,
    registrar_en_bitacora,
    tabla_folds,
)


@pytest.fixture
def datos() -> tuple[pd.DataFrame, pd.Series]:
    """Datos sintéticos con señal: el objetivo depende de ``x1``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=600), "x2": rng.normal(size=600)})
    y = pd.Series((X["x1"] + rng.normal(scale=1.0, size=600) > 1).astype(int))
    return X, y


@pytest.fixture
def cv() -> StratifiedKFold:
    """Partición pequeña y reproducible."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=1)


def test_oof_cubre_todas_las_filas(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    resultado = evaluar_modelo(LogisticRegression(), X, y, "logistica", cv=cv)
    assert not np.isnan(resultado.oof).any()
    assert len(resultado.auc_folds) == 3
    assert 0.5 < resultado.auc_oof <= 1.0
    assert resultado.auc_train >= resultado.auc_media - 0.05


def test_cada_fold_usa_un_clon_nuevo(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    modelo = make_pipeline(StandardScaler(), LogisticRegression())
    evaluar_modelo(modelo, X, y, "pipeline", cv=cv)
    # El modelo original no se ajusta: cada fold trabaja con un clon
    assert not hasattr(modelo[-1], "coef_")


def test_early_stopping_registra_arboles_por_fold(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    modelo = make_pipeline(
        StandardScaler(),
        XGBClassifier(n_estimators=500, learning_rate=0.3, max_depth=2, random_state=0),
    )
    resultado = evaluar_modelo(modelo, X, y, "xgb", cv=cv, early_stopping_rounds=10)
    assert len(resultado.n_arboles) == 3
    assert all(1 <= n <= 500 for n in resultado.n_arboles)


def test_early_stopping_falla_con_modelos_sin_soporte(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    with pytest.raises((ValueError, TypeError)):
        evaluar_modelo(
            LogisticRegression(), X, y, "logistica", cv=cv, early_stopping_rounds=5
        )


def test_valida_objetivo_binario(datos: tuple[pd.DataFrame, pd.Series]) -> None:
    X, _ = datos
    with pytest.raises(ValueError, match="binario"):
        evaluar_modelo(LogisticRegression(), X, pd.Series([2] * len(X)), "x")


def test_tabla_de_folds_incluye_promedio_y_desviacion(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    tabla = tabla_folds(evaluar_modelo(LogisticRegression(), X, y, "lr", cv=cv))
    assert list(tabla.index[-2:]) == ["promedio", "desv"]
    assert tabla.loc["promedio", "AUC validación"] == pytest.approx(
        tabla.iloc[:3]["AUC validación"].mean()
    )


def test_registrar_reemplaza_el_experimento_repetido(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold, tmp_path: Path
) -> None:
    X, y = datos
    bitacora = tmp_path / "experiments.csv"
    resultado = evaluar_modelo(LogisticRegression(), X, y, "jae_prueba", cv=cv)
    registrar_en_bitacora(resultado, "x1, x2", "logística", log_path=bitacora)
    registrar_en_bitacora(resultado, "x1, x2", "logística", log_path=bitacora)
    assert (read_log(bitacora)["nombre"] == "jae_prueba").sum() == 1


def test_registrar_conserva_el_puntaje_de_kaggle(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold, tmp_path: Path
) -> None:
    X, y = datos
    bitacora = tmp_path / "experiments.csv"
    resultado = evaluar_modelo(LogisticRegression(), X, y, "jae_prueba", cv=cv)
    registrar_en_bitacora(resultado, "x1, x2", "logística", log_path=bitacora)
    anotada = read_log(bitacora)
    anotada.loc[anotada["nombre"] == "jae_prueba", "kaggle_score"] = 0.88
    anotada.to_csv(bitacora, index=False)
    registrar_en_bitacora(resultado, "x1, x2", "logística", log_path=bitacora)
    fila = read_log(bitacora).query("nombre == 'jae_prueba'")
    assert len(fila) == 1
    assert fila["kaggle_score"].iloc[0] == pytest.approx(0.88)


def test_combinar_promedia_las_probabilidades_oof(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    a = evaluar_modelo(LogisticRegression(C=0.01), X, y, "a", cv=cv)
    b = evaluar_modelo(LogisticRegression(C=10.0), X, y, "b", cv=cv)
    mezcla = combinar_resultados([a, b], y, "mezcla", cv=cv)
    assert np.allclose(mezcla.oof, (a.oof + b.oof) / 2)
    assert len(mezcla.auc_folds) == 3
    assert mezcla.tiempo_s == pytest.approx(a.tiempo_s + b.tiempo_s)


def test_combinar_valida_los_pesos(
    datos: tuple[pd.DataFrame, pd.Series], cv: StratifiedKFold
) -> None:
    X, y = datos
    a = evaluar_modelo(LogisticRegression(), X, y, "a", cv=cv)
    with pytest.raises(ValueError, match="peso"):
        combinar_resultados([a], y, "mezcla", cv=cv, pesos=[0.5, 0.5])
    with pytest.raises(ValueError, match="al menos"):
        combinar_resultados([], y, "mezcla", cv=cv)
