"""Ingeniería de variables (propuesta independiente JAE) para la competencia crediticia.

El módulo separa dos tipos de transformación:

- **Fijas (deterministas):** :func:`clean_raw` y :func:`add_fixed_features`. No
  aprenden nada de los datos, así que se pueden aplicar antes de la validación
  cruzada.
- **Aprendidas:** :class:`GroupMedianImputer`, :class:`WoEEncoder` y los
  escaladores. Se ajustan solo con el pliegue de entrenamiento dentro del
  ``Pipeline`` que construye :func:`build_preprocessor`.

Los umbrales fijos provienen del EDA (``notebooks/02_eda.ipynb``, sección 10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# --- Nombres de columnas originales ---------------------------------------
TARGET = "SeriousDlqin2yrs"
ID = "Id"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
AGE = "age"
LATE30 = "NumberOfTime30-59DaysPastDueNotWorse"
LATE60 = "NumberOfTime60-89DaysPastDueNotWorse"
LATE90 = "NumberOfTimes90DaysLate"
LATE_COLS = [LATE30, LATE60, LATE90]
DEBT_RATIO = "DebtRatio"
INCOME = "MonthlyIncome"
OPEN_LINES = "NumberOfOpenCreditLinesAndLoans"
REAL_ESTATE = "NumberRealEstateLoansOrLines"
DEPENDENTS = "NumberOfDependents"
RAW_FEATURES = [
    UTIL,
    AGE,
    LATE30,
    DEBT_RATIO,
    INCOME,
    OPEN_LINES,
    LATE90,
    REAL_ESTATE,
    LATE60,
    DEPENDENTS,
]

# --- Umbrales fijos (EDA, sección 10) --------------------------------------
LATE_SENTINEL_MIN = 96
MIN_ADULT_AGE = 18
AGE_BINS = (0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120)
UTIL_MAX = 2.0
LATE_MAX = 6
DEBT_RATIO_VALID_MAX = 10.0
INCOME_MAX = 25_000.0
OPEN_LINES_MAX = 24
REAL_ESTATE_MAX = 5
DEPENDENTS_MAX = 5
MORTGAGE_LEVEL_MAX = 2

# --- Nombres de variables construidas --------------------------------------
LATE_SHORT = {LATE30: "late_30_59", LATE60: "late_60_89", LATE90: "late_90"}
BASE_FLAGS = [
    "late_sentinel",
    "income_missing",
    "income_zero",
    "dependents_missing",
    "util_gt1",
]
PROFILE_FLAGS = ["util_zero", "thin_file"]
INCOME_OUTPUTS = {
    "global": "income_imp_global",
    "A": "income_imp_age_mortgage",
    "B": "income_imp_age_debt",
}

IncomeStrategy = Literal["global", "A", "B", "AB"]
ScalerName = Literal["standard", "robust", "quantile"]
LateMode = Literal["count", "binary"]
Representation = Literal["scaled", "woe", "splines"]


def _require_columns(df: pd.DataFrame, columns: list[str], step: str) -> None:
    """Valida que existan columnas requeridas.

    Args:
        df: DataFrame a validar.
        columns: Columnas que deben existir.
        step: Nombre del paso, para el mensaje de error.

    Raises:
        ValueError: Si falta alguna columna.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.error("Faltan columnas en %s: %s", step, missing)
        raise ValueError(f"Faltan columnas en {step}: {missing}")


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza fija a los datos crudos.

    - Códigos 96/98 de atrasos → ``NaN`` con indicador ``late_sentinel``.
    - Edad menor a 18 → ``NaN``.
    - Indicadores de ingreso faltante, ingreso cero y dependientes faltantes.

    Args:
        df: Datos crudos con las columnas originales de la competencia.

    Returns:
        Copia limpia del DataFrame con los indicadores agregados.

    Raises:
        ValueError: Si faltan columnas originales.
    """
    _require_columns(df, RAW_FEATURES, "clean_raw")
    out = df.copy()
    sentinel = (out[LATE_COLS] >= LATE_SENTINEL_MIN).any(axis=1)
    out["late_sentinel"] = sentinel.astype(int)
    out[LATE_COLS] = out[LATE_COLS].where(out[LATE_COLS] < LATE_SENTINEL_MIN)
    out[AGE] = out[AGE].where(out[AGE] >= MIN_ADULT_AGE)
    out["income_missing"] = out[INCOME].isna().astype(int)
    out["income_zero"] = (out[INCOME] == 0).astype(int)
    out["dependents_missing"] = out[DEPENDENTS].isna().astype(int)
    logger.info(
        "Limpieza fija: %d filas, %d con códigos 96/98, %d edades inválidas",
        len(out),
        int(sentinel.sum()),
        int(out[AGE].isna().sum()),
    )
    return out


def add_fixed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables fijas (deterministas) sobre datos ya limpios.

    Args:
        df: Salida de :func:`clean_raw`.

    Returns:
        Copia del DataFrame con las variables construidas.

    Raises:
        ValueError: Si faltan columnas de la limpieza fija.
    """
    _require_columns(df, RAW_FEATURES + BASE_FLAGS[:4], "add_fixed_features")
    out = df.copy()
    util = out[UTIL]
    out["util_gt1"] = (util > 1).astype(int)
    out["util_zero"] = (util == 0).astype(int)
    out["util_clip"] = util.clip(0, UTIL_MAX)

    # Atrasos: los códigos 96/98 ya son NaN; se llevan a 0 (el indicador los conserva)
    late = out[LATE_COLS].fillna(0).clip(upper=LATE_MAX)
    for col, short in LATE_SHORT.items():
        out[short] = late[col]
        out[f"{short}_any"] = (late[col] > 0).astype(int)
    out["total_late"] = late.sum(axis=1)
    out["any_late"] = (out["total_late"] > 0).astype(int)
    out["late_severity"] = late[LATE30] + 2 * late[LATE60] + 3 * late[LATE90]

    out["thin_file"] = (
        (out["total_late"] == 0)
        & (out["late_sentinel"] == 0)
        & (util == 0)
        & (out[OPEN_LINES] <= 1)
    ).astype(int)

    # Sin ingreso (o ingreso 0), DebtRatio es el monto mensual de deuda
    no_income = out[INCOME].isna() | (out[INCOME] <= 0)
    out["debt_amount"] = out[DEBT_RATIO].where(no_income, out[DEBT_RATIO] * out[INCOME])
    out["debt_ratio_valid"] = (
        out[DEBT_RATIO].where(~no_income).clip(upper=DEBT_RATIO_VALID_MAX).fillna(0)
    )

    out["income_clip"] = out[INCOME].clip(upper=INCOME_MAX)
    out["open_lines"] = out[OPEN_LINES].clip(upper=OPEN_LINES_MAX)
    out["real_estate"] = out[REAL_ESTATE].clip(upper=REAL_ESTATE_MAX)
    out["dependents"] = out[DEPENDENTS].clip(upper=DEPENDENTS_MAX)

    # Llaves de grupo para la imputación (la edad inválida va al grupo más joven)
    out["age_group"] = (
        pd.cut(out[AGE], AGE_BINS, labels=False, include_lowest=True)
        .fillna(0)
        .astype(int)
    )
    out["mortgage_level"] = out[REAL_ESTATE].clip(upper=MORTGAGE_LEVEL_MAX).astype(int)
    logger.info("Variables fijas agregadas: %d columnas en total", out.shape[1])
    return out


def prepare_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza y las variables fijas en un solo paso.

    Args:
        df: Datos crudos.

    Returns:
        Datos listos para el ``Pipeline`` de :func:`build_preprocessor`.
    """
    return add_fixed_features(clean_raw(df))


def quantile_edges(values: pd.Series, n_bins: int) -> np.ndarray:
    """Calcula cortes por cuantiles que también funcionan con variables discretas.

    Usa solo los cuantiles interiores y abre los extremos a ±infinito. Así, en
    una variable con 90 % de ceros los cortes quedan en ``[-inf, 0, ..., inf]``
    y el cero conserva su propio rango, en lugar de fundirse con todo lo demás.

    Args:
        values: Valores observados (sin faltantes).
        n_bins: Número máximo de rangos.

    Returns:
        Cortes ordenados, de ``-inf`` a ``inf``.

    Raises:
        ValueError: Si no hay valores o ``n_bins`` es menor a 2.
    """
    if n_bins < 2:
        raise ValueError("n_bins debe ser al menos 2")
    if len(values) == 0:
        raise ValueError("No hay valores para calcular los cortes")
    inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa una columna con la mediana de su grupo, aprendida en ``fit``.

    Los grupos se forman con ``group_cols`` y, opcionalmente, con los cuantiles
    de ``binned_col`` (cortes aprendidos también en ``fit``). Si un grupo no
    existe en el ajuste, se usa la mediana global.

    Args:
        source_col: Columna con faltantes a imputar.
        output_col: Columna de salida (puede ser la misma que ``source_col``).
        group_cols: Columnas categóricas que definen los grupos.
        binned_col: Columna numérica que se corta en cuantiles para agrupar.
        n_bins: Número de cuantiles de ``binned_col``.
        positive_only: Si es ``True``, solo los valores > 0 entran al ajuste.
    """

    def __init__(
        self,
        source_col: str,
        output_col: str,
        group_cols: tuple[str, ...] = (),
        binned_col: str | None = None,
        n_bins: int = 10,
        positive_only: bool = False,
    ) -> None:
        self.source_col = source_col
        self.output_col = output_col
        self.group_cols = group_cols
        self.binned_col = binned_col
        self.n_bins = n_bins
        self.positive_only = positive_only

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        """Construye la llave de grupo de cada fila."""
        parts = [X[c].astype(str) for c in self.group_cols]
        if self.binned_col is not None:
            bins = pd.cut(X[self.binned_col], self.bin_edges_, labels=False)
            parts.append(bins.astype("Int64").astype(str))
        if not parts:
            return pd.Series("todos", index=X.index)
        key = parts[0]
        for part in parts[1:]:
            key = key + "_" + part
        return key

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> GroupMedianImputer:
        """Aprende las medianas por grupo.

        Args:
            X: Datos del pliegue de entrenamiento.
            y: Ignorado; se acepta por compatibilidad con ``Pipeline``.

        Returns:
            El propio imputador ajustado.

        Raises:
            ValueError: Si faltan columnas o no hay valores para ajustar.
        """
        cols = [self.source_col, *self.group_cols]
        if self.binned_col is not None:
            cols.append(self.binned_col)
        _require_columns(X, cols, type(self).__name__)
        source = X[self.source_col]
        mask = source.notna() & (source > 0 if self.positive_only else True)
        if not mask.any():
            raise ValueError(
                f"No hay valores para ajustar la imputación de {self.source_col}"
            )
        if self.binned_col is not None:
            self.bin_edges_ = quantile_edges(X.loc[mask, self.binned_col], self.n_bins)
        keys = self._keys(X)
        self.group_medians_ = source[mask].groupby(keys[mask]).median().to_dict()
        self.global_median_ = float(source[mask].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputa los faltantes con la mediana del grupo.

        Args:
            X: Datos a transformar.

        Returns:
            Copia de ``X`` con ``output_col`` sin faltantes.
        """
        check_is_fitted(self, "group_medians_")
        out = X.copy()
        imputed = (
            self._keys(out)
            .map(self.group_medians_)
            .astype(float)
            .fillna(self.global_median_)
        )
        missing = out[self.source_col].isna()
        out[self.output_col] = out[self.source_col].where(~missing, imputed)
        logger.debug("%s: %d valores imputados", self.output_col, int(missing.sum()))
        return out


def add_disposable_income(df: pd.DataFrame, income_cols: list[str]) -> pd.DataFrame:
    """Agrega el ingreso disponible (ingreso − monto mensual de deuda).

    Args:
        df: Datos con ``debt_amount`` y las columnas de ingreso indicadas.
        income_cols: Columnas de ingreso a partir de las cuales se calcula.

    Returns:
        Copia con una columna ``disposable_<ingreso>`` por cada ingreso.
    """
    _require_columns(df, [*income_cols, "debt_amount"], "add_disposable_income")
    out = df.copy()
    for col in income_cols:
        out[f"disposable_{col}"] = out[col] - out["debt_amount"]
    return out


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Logaritmo con signo: ``sign(x) · log1p(|x|)``, para valores negativos.

    Args:
        x: Arreglo numérico.

    Returns:
        Arreglo transformado.
    """
    return np.sign(x) * np.log1p(np.abs(x))


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada variable por el WoE de su rango (cuantiles + rango de faltantes).

    El WoE de un rango es ``ln(% no morosos / % morosos)``: negativo indica más
    riesgo. Los cortes y el WoE se aprenden en ``fit`` con la variable objetivo,
    por eso deben ajustarse dentro de cada pliegue.

    Args:
        columns: Columnas a codificar.
        n_bins: Número máximo de cuantiles por variable.
        smoothing: Valor sumado a cada conteo para evitar divisiones entre cero.
    """

    def __init__(
        self, columns: list[str], n_bins: int = 10, smoothing: float = 0.5
    ) -> None:
        self.columns = columns
        self.n_bins = n_bins
        self.smoothing = smoothing

    def _bins(self, values: pd.Series, edges: np.ndarray) -> pd.Series:
        """Asigna cada valor a su rango; los faltantes van al rango -1."""
        return pd.cut(values, edges, labels=False).astype(float).fillna(-1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoEEncoder:
        """Aprende cortes, WoE e IV de cada variable.

        Args:
            X: Datos del pliegue de entrenamiento.
            y: Variable objetivo binaria (1 = moroso).

        Returns:
            El propio codificador ajustado.

        Raises:
            ValueError: Si faltan columnas o ``y`` no es binaria.
        """
        _require_columns(X, list(self.columns), type(self).__name__)
        target = pd.Series(np.asarray(y), index=X.index)
        if set(target.unique()) - {0, 1}:
            raise ValueError("WoEEncoder requiere una variable objetivo binaria 0/1")
        self.edges_, self.woe_, self.iv_ = {}, {}, {}
        for col in self.columns:
            observed = X[col].dropna()
            edges = quantile_edges(observed, self.n_bins)
            bins = self._bins(X[col], edges)
            counts = pd.crosstab(bins, target).reindex(columns=[0, 1], fill_value=0)
            counts = counts + self.smoothing
            good = counts[0] / counts[0].sum()
            bad = counts[1] / counts[1].sum()
            woe = np.log(good / bad)
            self.edges_[col] = edges
            self.woe_[col] = woe.to_dict()
            self.iv_[col] = float(((good - bad) * woe).sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza cada variable por su WoE.

        Args:
            X: Datos a transformar.

        Returns:
            DataFrame con columnas ``woe_<variable>``. Un rango no visto en el
            ajuste recibe WoE 0 (sin evidencia).
        """
        check_is_fitted(self, "woe_")
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            bins = self._bins(X[col], self.edges_[col])
            out[f"woe_{col}"] = bins.map(self.woe_[col]).astype(float).fillna(0.0)
        return out

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        """Nombres de las columnas de salida."""
        return np.array([f"woe_{c}" for c in self.columns])


@dataclass(frozen=True)
class FeatureConfig:
    """Configuración de un experimento de ingeniería de variables.

    Attributes:
        income_strategy: Imputación del ingreso: ``global``, ``A`` (edad +
            hipotecas), ``B`` (edad + decil de deuda) o ``AB`` (ambas).
        scaler: Escalador para modelos lineales.
        late_mode: Atrasos como conteo transformado o como indicadores.
        representation: ``scaled`` (variables escaladas), ``woe`` o ``splines``.
        disposable_income: Si se agrega el ingreso disponible.
        profile_flags: Si se agregan ``util_zero`` y ``thin_file``.
        for_trees: Si el preprocesamiento es para modelos de árboles (sin
            escalado).
        impute_for_trees: En árboles, si se imputa o se dejan los ``NaN``.
    """

    income_strategy: IncomeStrategy = "A"
    scaler: ScalerName = "standard"
    late_mode: LateMode = "count"
    representation: Representation = "scaled"
    disposable_income: bool = True
    profile_flags: bool = True
    for_trees: bool = False
    impute_for_trees: bool = True


def _income_imputers(strategy: IncomeStrategy) -> list[tuple[str, GroupMedianImputer]]:
    """Construye los imputadores de ingreso de la estrategia indicada."""
    specs = {
        "global": GroupMedianImputer(
            "income_clip", INCOME_OUTPUTS["global"], positive_only=True
        ),
        "A": GroupMedianImputer(
            "income_clip",
            INCOME_OUTPUTS["A"],
            group_cols=("age_group", "mortgage_level"),
            positive_only=True,
        ),
        "B": GroupMedianImputer(
            "income_clip",
            INCOME_OUTPUTS["B"],
            group_cols=("age_group",),
            binned_col="debt_amount",
            positive_only=True,
        ),
    }
    names = ["A", "B"] if strategy == "AB" else [strategy]
    return [(f"income_{n}", specs[n]) for n in names]


def income_columns(strategy: IncomeStrategy) -> list[str]:
    """Columnas de ingreso imputado que produce una estrategia.

    Args:
        strategy: Estrategia de imputación del ingreso.

    Returns:
        Lista de columnas de ingreso imputado.
    """
    names = ["A", "B"] if strategy == "AB" else [strategy]
    return [INCOME_OUTPUTS[n] for n in names]


def _scaler(name: ScalerName) -> BaseEstimator:
    """Devuelve el escalador indicado."""
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    raise ValueError(f"Escalador desconocido: {name}")


def feature_groups(config: FeatureConfig) -> dict[str, list[str]]:
    """Agrupa las columnas finales según el tratamiento que reciben.

    Args:
        config: Configuración del experimento.

    Returns:
        Diccionario con las listas ``log``, ``signed_log``, ``plain`` y ``flags``.
    """
    incomes = income_columns(config.income_strategy)
    late_count = list(LATE_SHORT.values())
    late_binary = [f"{c}_any" for c in late_count]
    groups = {
        "log": [*incomes, "debt_amount", "debt_ratio_valid"],
        "signed_log": (
            [f"disposable_{c}" for c in incomes] if config.disposable_income else []
        ),
        "plain": ["util_clip", AGE, "open_lines", "real_estate", "dependents"],
        "flags": [*BASE_FLAGS, *(PROFILE_FLAGS if config.profile_flags else [])],
    }
    if config.late_mode == "count":
        groups["log"] += late_count
    else:
        groups["flags"] += late_binary
    return groups


def _pandas_pipeline(steps: list[tuple[str, object]]) -> Pipeline:
    """Arma el ``Pipeline`` con salida pandas en los ``ColumnTransformer``.

    Los transformadores propios ya devuelven ``DataFrame``; solo los de
    ``scikit-learn`` necesitan configurarse.

    Args:
        steps: Pasos del ``Pipeline``.

    Returns:
        ``Pipeline`` cuyos pasos producen ``DataFrame``.
    """
    for _, step in steps:
        if isinstance(step, (ColumnTransformer, FunctionTransformer)):
            step.set_output(transform="pandas")
    return Pipeline(steps)


def build_preprocessor(config: FeatureConfig) -> Pipeline:
    """Construye el ``Pipeline`` de transformaciones aprendidas.

    Recibe la salida de :func:`prepare_fixed`. Todos los pasos se ajustan solo
    con los datos que recibe ``fit`` (el pliegue de entrenamiento).

    Args:
        config: Configuración del experimento.

    Returns:
        ``Pipeline`` listo para combinarse con un modelo.
    """
    impute = not config.for_trees or config.impute_for_trees
    steps: list[tuple[str, object]] = []
    if impute:
        steps += _income_imputers(config.income_strategy)
        steps.append(
            (
                "dependents",
                GroupMedianImputer(
                    "dependents", "dependents", group_cols=("age_group",)
                ),
            )
        )
    groups = feature_groups(config)

    if not impute:
        # Árboles sin imputar: ingreso con NaN; el modelo los maneja directamente
        incomes = income_columns(config.income_strategy)
        groups["log"] = ["income_clip", *[c for c in groups["log"] if c not in incomes]]
        groups["signed_log"] = []
        if config.disposable_income:
            steps.append(
                (
                    "disposable",
                    FunctionTransformer(
                        add_disposable_income, kw_args={"income_cols": ["income_clip"]}
                    ),
                )
            )
            groups["signed_log"] = ["disposable_income_clip"]
    elif config.disposable_income:
        steps.append(
            (
                "disposable",
                FunctionTransformer(
                    add_disposable_income,
                    kw_args={"income_cols": income_columns(config.income_strategy)},
                ),
            )
        )

    if config.for_trees:
        columns = [c for cols in groups.values() for c in cols] + [
            "total_late",
            "late_severity",
        ]
        steps.append(
            (
                "select",
                ColumnTransformer(
                    [("keep", "passthrough", columns)],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            )
        )
        return _pandas_pipeline(steps)

    if config.representation == "woe":
        woe_cols = [
            UTIL,
            AGE,
            *LATE_COLS,
            "debt_ratio_valid",
            "debt_amount",
            INCOME,
            OPEN_LINES,
            REAL_ESTATE,
            DEPENDENTS,
        ]
        if config.disposable_income:
            woe_cols.append(groups["signed_log"][0])
        steps.append(
            (
                "final",
                ColumnTransformer(
                    [
                        ("woe", WoEEncoder(woe_cols), woe_cols),
                        ("flags", "passthrough", groups["flags"]),
                    ],
                    verbose_feature_names_out=False,
                ),
            )
        )
        return _pandas_pipeline(steps)

    branches = [
        (
            "log",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "log",
                        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    ),
                    ("scale", _scaler(config.scaler)),
                ]
            ),
            groups["log"],
        ),
        (
            "plain",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", _scaler(config.scaler)),
                ]
            ),
            groups["plain"],
        ),
        ("flags", "passthrough", groups["flags"]),
    ]
    if groups["signed_log"]:
        branches.append(
            (
                "signed_log",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        (
                            "log",
                            FunctionTransformer(
                                signed_log1p, feature_names_out="one-to-one"
                            ),
                        ),
                        ("scale", _scaler(config.scaler)),
                    ]
                ),
                groups["signed_log"],
            )
        )
    if config.representation == "splines":
        branches.append(
            (
                "splines",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("spline", SplineTransformer(n_knots=5, degree=3)),
                    ]
                ),
                ["util_clip", AGE],
            )
        )
    steps.append(
        ("final", ColumnTransformer(branches, verbose_feature_names_out=False))
    )
    return _pandas_pipeline(steps)
