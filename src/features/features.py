"""Features deterministas (sin estadísticos aprendidos): seguras dentro de cualquier CV."""
import numpy as np
import pandas as pd

TARGET, ID = "SeriousDlqin2yrs", "Id"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
LATE30 = "NumberOfTime30-59DaysPastDueNotWorse"
LATE60 = "NumberOfTime60-89DaysPastDueNotWorse"
LATE90 = "NumberOfTimes90DaysLate"
LATE_COLS = [LATE30, LATE60, LATE90]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["late_sentinel"] = (df[LATE_COLS] >= 96).any(axis=1).astype(int)
    df[LATE_COLS] = df[LATE_COLS].where(df[LATE_COLS] < 96)
    df["age"] = df["age"].where(df["age"] >= 18)
    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["income_zero"] = (df["MonthlyIncome"] == 0).astype(int)
    df["dependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    late = df[LATE_COLS]
    inc = df["MonthlyIncome"]
    util = df[UTIL]
    no_inc = inc.isna() | (inc <= 0)

    # Atrasos agregados
    df["total_late"] = late.sum(axis=1, min_count=1)
    df["max_late"] = late.max(axis=1)
    df["any_late"] = (late.fillna(0) > 0).any(axis=1).astype(int)
    df["late_severity"] = (
        late[LATE30].fillna(0) + 2 * late[LATE60].fillna(0) + 3 * late[LATE90].fillna(0)
    )

    # Utilización
    df["util_clip"] = util.clip(upper=2)
    df["util_gt1"] = (util > 1).astype(int)

    # Ingreso y deuda. Hipótesis (EDA): con ingreso faltante o 0, DebtRatio se comporta
    # como un monto y no como una razón.
    df["log_income"] = np.log1p(inc)
    df["debt_ratio_valid"] = df["DebtRatio"].where(~no_inc).clip(upper=10)
    df["debt_amount"] = df["DebtRatio"].where(no_inc, df["DebtRatio"] * inc)
    df["log_debt_amount"] = np.log1p(df["debt_amount"])
    df["income_per_dep"] = inc / (1 + df["NumberOfDependents"])

    # Estructura de crédito e interacciones
    df["non_re_lines"] = df["NumberOfOpenCreditLinesAndLoans"] - df["NumberRealEstateLoansOrLines"]
    df["age_x_util"] = df["age"] * df["util_clip"]
    df["late_x_util"] = df["total_late"].fillna(0) * df["util_clip"]
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[TARGET, ID], errors="ignore")
    return add_features(clean(df))


FLAGS = ["late_sentinel", "income_missing", "income_zero", "dependents_missing"]
MIN_EXTRA = ["total_late", "any_late"]


def prepare_min(df: pd.DataFrame) -> pd.DataFrame:
    """Set candidato mínimo (16 columnas): originales limpias + banderas + total_late + any_late."""
    full = prepare(df)
    raw_cols = [c for c in df.columns if c not in (TARGET, ID)]
    return full[raw_cols + FLAGS + MIN_EXTRA]
