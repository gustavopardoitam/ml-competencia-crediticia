"""Validación cruzada con predicciones fuera de fold (OOF) para comparar modelos.

Complementa a :func:`evaluation.cv.evaluate`: usa la misma partición del equipo
(:func:`evaluation.cv.get_cv`), pero además de la media y la desviación del AUC
devuelve el AUC de entrenamiento, las predicciones OOF y las curvas ROC de cada
fold. Soporta *early stopping* para modelos de *boosting* sin usar el fold
de validación para elegir el número de árboles.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from evaluation.cv import RANDOM_STATE, get_cv
from evaluation.log import LOG_PATH, log_experiment, read_log

logger = logging.getLogger(__name__)


@dataclass
class ResultadoCV:
    """Resultado de evaluar un modelo con validación cruzada.

    Attributes:
        nombre: Nombre del experimento.
        auc_folds: AUC de validación de cada fold.
        auc_train_folds: AUC sobre los datos de entrenamiento de cada fold.
        oof: Probabilidad predicha para cada fila cuando estuvo en validación.
        auc_oof: AUC global calculado con todas las predicciones OOF.
        tiempo_s: Segundos que tomó la validación cruzada completa.
        curvas: Pares ``(y_validacion, probabilidad)`` de cada fold.
        n_arboles: Árboles elegidos por *early stopping* en cada fold
            (vacío si no se usó).
    """

    nombre: str
    auc_folds: np.ndarray
    auc_train_folds: np.ndarray
    oof: np.ndarray
    auc_oof: float
    tiempo_s: float
    curvas: list[tuple[np.ndarray, np.ndarray]] = field(repr=False)
    n_arboles: list[int] = field(default_factory=list)

    @property
    def auc_media(self) -> float:
        """AUC de validación promedio entre folds."""
        return float(self.auc_folds.mean())

    @property
    def auc_std(self) -> float:
        """Desviación estándar del AUC de validación entre folds."""
        return float(self.auc_folds.std())

    @property
    def auc_train(self) -> float:
        """AUC de entrenamiento promedio entre folds."""
        return float(self.auc_train_folds.mean())

    @property
    def brecha(self) -> float:
        """Diferencia entre el AUC de entrenamiento y el de validación."""
        return self.auc_train - self.auc_media

    def resumen(self) -> dict[str, float | str]:
        """Resume el experimento en una fila para la tabla comparativa.

        Returns:
            Diccionario con las métricas principales del experimento.
        """
        return {
            "experimento": self.nombre,
            "AUC entrenamiento": self.auc_train,
            "AUC validación (media)": self.auc_media,
            "AUC validación (desv)": self.auc_std,
            "AUC OOF global": self.auc_oof,
            "brecha": self.brecha,
            "tiempo (s)": self.tiempo_s,
        }


def _ajustar_con_parada(
    modelo: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    rondas: int,
    fraccion_parada: float,
    semilla: int,
) -> int:
    """Ajusta un modelo de *boosting* con *early stopping* sobre una partición interna.

    El conjunto de parada se separa del fold de **entrenamiento**, nunca del de
    validación, para que el número de árboles no se elija viendo los datos con
    los que después se mide el AUC.

    Args:
        modelo: Estimador o ``Pipeline`` cuyo último paso acepta ``eval_set`` y
            expone ``best_iteration`` (por ejemplo, ``XGBClassifier``).
        X: Variables del fold de entrenamiento.
        y: Objetivo del fold de entrenamiento.
        rondas: Rondas sin mejora antes de detener el entrenamiento.
        fraccion_parada: Fracción del fold que se reserva para la parada.
        semilla: Semilla de la partición interna.

    Returns:
        Número de árboles elegido.

    Raises:
        ValueError: Si el último paso del modelo no reporta ``best_iteration``.
    """
    X_aj, X_par, y_aj, y_par = train_test_split(
        X, y, test_size=fraccion_parada, stratify=y, random_state=semilla
    )
    if isinstance(modelo, Pipeline):
        preprocesador, final = modelo[:-1], modelo[-1]
        X_aj = preprocesador.fit_transform(X_aj, y_aj)
        X_par = preprocesador.transform(X_par)
    else:
        final = modelo
    final.set_params(early_stopping_rounds=rondas)
    final.fit(X_aj, y_aj, eval_set=[(X_par, y_par)], verbose=False)
    mejor = getattr(final, "best_iteration", None)
    if mejor is None:
        logger.error("%s no reporta best_iteration", type(final).__name__)
        raise ValueError("El modelo no admite early stopping con eval_set")
    return int(mejor) + 1


def evaluar_modelo(
    modelo: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    nombre: str,
    cv: StratifiedKFold | None = None,
    early_stopping_rounds: int | None = None,
    fraccion_parada: float = 0.1,
    semilla: int = RANDOM_STATE,
) -> ResultadoCV:
    """Evalúa un modelo con validación cruzada y guarda las predicciones OOF.

    Todo lo que el modelo aprende (imputaciones, escalado, árboles) se ajusta
    solo con el fold de entrenamiento, porque en cada fold se usa un clon
    nuevo del modelo.

    Args:
        modelo: Estimador o ``Pipeline`` con ``fit`` y ``predict_proba``.
        X: Variables de entrada.
        y: Objetivo binario (1 = moroso).
        nombre: Nombre del experimento.
        cv: Partición; por omisión, la compartida del equipo (``get_cv()``).
        early_stopping_rounds: Si se indica, el último paso se ajusta con
            *early stopping* sobre una partición interna del fold de
            entrenamiento.
        fraccion_parada: Fracción del fold de entrenamiento reservada para
            el *early stopping*.
        semilla: Semilla de la partición interna.

    Returns:
        Resultado con AUC por fold, AUC de entrenamiento, AUC OOF global,
        predicciones OOF y curvas ROC.

    Raises:
        ValueError: Si ``y`` no es binaria o ``X`` y ``y`` no tienen el mismo
            número de filas.
    """
    if len(X) != len(y):
        raise ValueError("X y y deben tener el mismo número de filas")
    if set(pd.unique(y)) - {0, 1}:
        raise ValueError("evaluar_modelo requiere un objetivo binario 0/1")
    cv = cv if cv is not None else get_cv()
    oof = np.full(len(y), np.nan)
    auc_val: list[float] = []
    auc_train: list[float] = []
    curvas: list[tuple[np.ndarray, np.ndarray]] = []
    arboles: list[int] = []
    inicio = time.perf_counter()
    for fold, (idx_tr, idx_va) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
        y_tr, y_va = y.iloc[idx_tr], y.iloc[idx_va]
        estimador = clone(modelo)
        if early_stopping_rounds is None:
            estimador.fit(X_tr, y_tr)
        else:
            arboles.append(
                _ajustar_con_parada(
                    estimador,
                    X_tr,
                    y_tr,
                    early_stopping_rounds,
                    fraccion_parada,
                    semilla,
                )
            )
        proba_va = estimador.predict_proba(X_va)[:, 1]
        oof[idx_va] = proba_va
        auc_val.append(roc_auc_score(y_va, proba_va))
        auc_train.append(roc_auc_score(y_tr, estimador.predict_proba(X_tr)[:, 1]))
        curvas.append((y_va.to_numpy(), proba_va))
        logger.info("%s · fold %d: AUC validación %.4f", nombre, fold, auc_val[-1])
    resultado = ResultadoCV(
        nombre=nombre,
        auc_folds=np.array(auc_val),
        auc_train_folds=np.array(auc_train),
        oof=oof,
        auc_oof=float(roc_auc_score(y, oof)),
        tiempo_s=time.perf_counter() - inicio,
        curvas=curvas,
        n_arboles=arboles,
    )
    logger.info(
        "%s: AUC %.4f ± %.4f (OOF %.4f)",
        nombre,
        resultado.auc_media,
        resultado.auc_std,
        resultado.auc_oof,
    )
    return resultado


def combinar_resultados(
    resultados: list[ResultadoCV],
    y: pd.Series,
    nombre: str,
    cv: StratifiedKFold | None = None,
    pesos: list[float] | None = None,
) -> ResultadoCV:
    """Combina varios experimentos promediando sus probabilidades OOF.

    Sirve para promediar semillas de un mismo modelo o para mezclar modelos
    distintos (*ensemble*). Como solo se promedian predicciones OOF ya
    calculadas, no hay data leakage: ninguna fila se predijo con un modelo que
    la haya visto.

    Args:
        resultados: Resultados de :func:`evaluar_modelo`, todos con la misma
            partición.
        y: Objetivo completo, en el mismo orden que las predicciones OOF.
        nombre: Nombre del experimento combinado.
        cv: Partición usada en los experimentos; por omisión, ``get_cv()``.
        pesos: Peso de cada experimento; por omisión, todos pesan igual.

    Returns:
        Resultado combinado. El AUC de entrenamiento es el promedio del de sus
        miembros y el tiempo es la suma.

    Raises:
        ValueError: Si no hay resultados o los pesos no coinciden en número.
    """
    if not resultados:
        raise ValueError("Se requiere al menos un resultado para combinar")
    if pesos is not None and len(pesos) != len(resultados):
        raise ValueError("Debe haber un peso por cada resultado")
    cv = cv if cv is not None else get_cv()
    oof = np.average([r.oof for r in resultados], axis=0, weights=pesos)
    auc_val: list[float] = []
    curvas: list[tuple[np.ndarray, np.ndarray]] = []
    for _, idx_va in cv.split(np.zeros(len(y)), y):
        y_va = y.iloc[idx_va].to_numpy()
        auc_val.append(roc_auc_score(y_va, oof[idx_va]))
        curvas.append((y_va, oof[idx_va]))
    return ResultadoCV(
        nombre=nombre,
        auc_folds=np.array(auc_val),
        auc_train_folds=np.mean([r.auc_train_folds for r in resultados], axis=0),
        oof=oof,
        auc_oof=float(roc_auc_score(y, oof)),
        tiempo_s=float(sum(r.tiempo_s for r in resultados)),
        curvas=curvas,
    )


def tabla_folds(resultado: ResultadoCV) -> pd.DataFrame:
    """Arma la tabla de AUC por fold con su promedio y desviación.

    Args:
        resultado: Resultado de :func:`evaluar_modelo`.

    Returns:
        Tabla con una fila por fold más las filas ``promedio`` y ``desv``.
    """
    tabla = pd.DataFrame(
        {
            "AUC validación": resultado.auc_folds,
            "AUC entrenamiento": resultado.auc_train_folds,
        },
        index=[f"fold {i}" for i in range(1, len(resultado.auc_folds) + 1)],
    )
    if resultado.n_arboles:
        tabla["n.º de árboles"] = resultado.n_arboles
    resumen = tabla.agg(["mean", "std"]).rename(
        index={"mean": "promedio", "std": "desv"}
    )
    resumen.loc["desv"] = tabla.std(ddof=0)
    return pd.concat([tabla, resumen])


def graficar_roc(
    resultado: ResultadoCV, ax: plt.Axes | None = None, titulo: str | None = None
) -> plt.Axes:
    """Grafica las curvas ROC de cada fold y la curva OOF global.

    Args:
        resultado: Resultado de :func:`evaluar_modelo`.
        ax: Ejes donde graficar; si es ``None`` se crean nuevos.
        titulo: Título de la gráfica; por omisión, el nombre del experimento.

    Returns:
        Ejes con la gráfica.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for i, (y_va, proba) in enumerate(resultado.curvas):
        fpr, tpr, _ = roc_curve(y_va, proba)
        ax.plot(fpr, tpr, color="lightgray", lw=1, label="folds" if i == 0 else None)
    # Las curvas de los folds juntas son exactamente las predicciones OOF
    y_total = np.concatenate([c[0] for c in resultado.curvas])
    proba_total = np.concatenate([c[1] for c in resultado.curvas])
    fpr, tpr, _ = roc_curve(y_total, proba_total)
    ax.plot(
        fpr,
        tpr,
        color="steelblue",
        lw=2.5,
        label=(
            f"OOF global (AUC {resultado.auc_oof:.4f}; "
            f"folds {resultado.auc_media:.4f} ± {resultado.auc_std:.4f})"
        ),
    )
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="azar (AUC 0.5)")
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de verdaderos positivos")
    ax.set_title(titulo or f"Curva ROC · {resultado.nombre}")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def graficar_roc_oof(
    resultados: list[ResultadoCV], y: pd.Series, ax: plt.Axes | None = None
) -> plt.Axes:
    """Compara en una sola gráfica las curvas ROC OOF de varios experimentos.

    Args:
        resultados: Resultados de :func:`evaluar_modelo` a comparar.
        y: Objetivo completo, en el mismo orden que las predicciones OOF.
        ax: Ejes donde graficar; si es ``None`` se crean nuevos.

    Returns:
        Ejes con la gráfica.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    for resultado in resultados:
        fpr, tpr, _ = roc_curve(y, resultado.oof)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{resultado.nombre} (AUC OOF {resultado.auc_oof:.4f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="azar (AUC 0.5)")
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de verdaderos positivos")
    ax.legend(loc="lower right", fontsize=9)
    return ax


def registrar_en_bitacora(
    resultado: ResultadoCV,
    features: str,
    modelo: str,
    notas: str = "",
    log_path: Path = LOG_PATH,
) -> pd.DataFrame:
    """Registra un experimento en la bitácora del equipo sin duplicarlo.

    Si ya existe una fila con el mismo nombre (por ejemplo, al volver a correr un
    notebook), se reemplaza en lugar de agregar otra y se conserva su
    ``kaggle_score``, que solo se anota a mano al subir la entrega.

    Args:
        resultado: Resultado de :func:`evaluar_modelo`.
        features: Descripción del conjunto de variables.
        modelo: Descripción del modelo.
        notas: Comentarios del experimento.
        log_path: Ruta de la bitácora.

    Returns:
        Bitácora completa después de registrar el experimento.
    """
    bitacora = read_log(log_path)
    kaggle_score: float | None = None
    previo = bitacora["nombre"] == resultado.nombre if not bitacora.empty else None
    if previo is not None and previo.any():
        logger.info("Se reemplaza el registro previo de %s", resultado.nombre)
        puntajes = bitacora.loc[previo, "kaggle_score"].dropna()
        kaggle_score = float(puntajes.iloc[-1]) if not puntajes.empty else None
        bitacora[~previo].to_csv(log_path, index=False)
    return log_experiment(
        nombre=resultado.nombre,
        features=features,
        modelo=modelo,
        cv_auc_mean=resultado.auc_media,
        cv_auc_std=resultado.auc_std,
        kaggle_score=kaggle_score,
        notas=notas,
        log_path=log_path,
    )
