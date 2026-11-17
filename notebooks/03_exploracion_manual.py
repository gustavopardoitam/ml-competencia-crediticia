# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # 03 · Exploración manual de variables
# Cada hipótesis se evalúa con ΔAUC pareado (mismos folds) frente a la base, con XGBoost fijo.
# Regla: si el ΔAUC no supera ~2 errores estándar, se descarta.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from features.features import ID, TARGET, prepare  # noqa: E402

train = pd.read_csv(ROOT / "data/raw/train.csv")
test = pd.read_csv(ROOT / "data/raw/test.csv")
y = train[TARGET].astype(int)
RAW = [c for c in train.columns if c not in (TARGET, ID)]
X_fe = prepare(train)

XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
           random_state=42, n_jobs=-1)
SPLITS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X_fe, y))
xgb_model = lambda: xgb.XGBClassifier(**XGB)


def fold_aucs(X, factory=xgb_model):
    return np.array([
        roc_auc_score(y.iloc[va], factory().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1])
        for tr, va in SPLITS
    ])


BASE = fold_aucs(X_fe)
results = [dict(variante="base (features actuales)", auc=BASE.mean(), delta=0.0, se=0.0, folds_mejores=None)]


def record(name, a):
    d = a - BASE
    results.append(dict(variante=name, auc=a.mean(), delta=d.mean(),
                        se=d.std(ddof=1) / np.sqrt(len(d)), folds_mejores=int((d > 0).sum())))
    return pd.DataFrame(results).round(5)


def compare(name, X_new):
    return record(name, fold_aucs(X_new))


print("AUC base (10 folds):", BASE.mean().round(5), "±", BASE.std().round(5))

# %% [markdown]
# ## 1. Mirada a mano: picos, valores exactos y sanidad del Id

# %%
u, inc = train["RevolvingUtilizationOfUnsecuredLines"], train["MonthlyIncome"]
masks = {"util == 0": u == 0, "0 < util < 0.05": (u > 0) & (u < 0.05), "util == 1": u == 1,
         "0.95 <= util < 1": (u >= 0.95) & (u < 1), "util > 1": u > 1}
for name, m in masks.items():
    print(f"{name:18s} n={m.sum():6d} ({m.mean():.1%})  tasa de impago={y[m].mean():.3f}")
r1000 = (inc > 0) & (inc % 1000 == 0)
r500 = (inc > 0) & (inc % 500 == 0) & ~r1000
rest = inc.notna() & (inc > 0) & ~r1000 & ~r500
for name, m in {"ingreso múltiplo de 1000": r1000, "múltiplo de 500": r500, "otro": rest}.items():
    print(f"{name:26s} n={m.sum():6d}  tasa de impago={y[m].mean():.3f}")
print("AUC de Id como predictor (esperado ~0.5):", round(roc_auc_score(y, train[ID]), 4))

# %% [markdown]
# ## 2. Familias de variables candidatas (sin usar el objetivo)

# %%
age_bin = pd.qcut(X_fe["age"], 10, labels=False, duplicates="drop")


def rel(c):
    return X_fe[c] - X_fe.groupby(age_bin)[c].transform("median")


cohort = pd.DataFrame({f"{c}_rel_age": rel(c) for c in
                       ["log_income", "util_clip", "NumberOfOpenCreditLinesAndLoans", "total_late"]})
cohort["income_pct_age"] = X_fe.groupby(age_bin)["log_income"].rank(pct=True)
cohort["util_pct_age"] = X_fe.groupby(age_bin)["util_clip"].rank(pct=True)

spikes = pd.DataFrame({
    "util_eq0": (u == 0).astype(int), "util_eq1": (u == 1).astype(int),
    "util_095_1": ((u >= 0.95) & (u <= 1)).astype(int),
    "inc_round1000": r1000.astype(int), "inc_round500": r500.astype(int)})

n_open = X_fe["NumberOfOpenCreditLinesAndLoans"]
late90 = X_fe["NumberOfTimes90DaysLate"].fillna(0)
late60 = X_fe["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
exposure = pd.DataFrame({
    "rev_exposure": X_fe["util_clip"] * n_open,
    "debt_per_line": X_fe["debt_amount"] / (1 + n_open),
    "income_per_line": X_fe["MonthlyIncome"] / (1 + n_open),
    "re_share": X_fe["NumberRealEstateLoansOrLines"] / (1 + n_open),
    "has_90": (late90 > 0).astype(int),
    "only_mild_late": ((X_fe["total_late"] > 0) & (late90 == 0) & (late60 == 0)).astype(int),
    "late90_share": X_fe["NumberOfTimes90DaysLate"] / X_fe["total_late"].where(X_fe["total_late"] > 0)})

miss = pd.DataFrame({"miss_pattern": X_fe["income_missing"] * 2 + X_fe["dependents_missing"]})
print(pd.crosstab(X_fe["income_missing"], X_fe["dependents_missing"]))

cnt = pd.concat([train["MonthlyIncome"], test["MonthlyIncome"]]).value_counts()
freq = pd.DataFrame({"income_freq": inc.map(cnt).fillna(0)})

# %%
families = {"+ cohorte por edad": cohort, "+ picos y redondez": spikes,
            "+ exposición y composición": exposure, "+ patrón de faltantes": miss,
            "+ frecuencia de ingreso (train+test)": freq}
for name, extra in families.items():
    out = compare(name, pd.concat([X_fe, extra], axis=1))
out

# %% [markdown]
# ## 3. Imputación: NaN nativo de XGBoost vs. mediana global

# %%
compare("imputación: mediana global", X_fe.fillna(X_fe.median()))

# %% [markdown]
# ## 4. Riesgo por segmento (análogo al ranking de lluvia), calculado dentro de cada fold
# Tasa de impago suavizada por celda (edad × utilización, atrasos × utilización).
# En entrenamiento se codifica con OOF interno; en validación, con todo el train del fold.

# %%
SEGS = {
    "seg_age_util": (("age", [17, 30, 40, 50, 60, 70, 110]), ("util_clip", [-0.01, 0.1, 0.3, 0.6, 1, 2])),
    "seg_late_util": (("total_late", [-1, 0, 1, 2, 100]), ("util_clip", [-0.01, 0.1, 0.3, 0.6, 1, 2])),
}


def _key(X, spec):
    (c1, b1), (c2, b2) = spec
    a = pd.cut(X[c1], b1, labels=False).fillna(-1).astype(int)
    b = pd.cut(X[c2], b2, labels=False).fillna(-1).astype(int)
    return a * 100 + b


def _enc(key_tr, y_tr, key_apply, alpha):
    prior = y_tr.mean()
    g = pd.DataFrame({"k": key_tr.values, "y": y_tr.values}).groupby("k")["y"].agg(["sum", "count"])
    rate = (g["sum"] + alpha * prior) / (g["count"] + alpha)
    return key_apply.map(rate).fillna(prior).values


def add_segment_risk(X_tr, y_tr, X_va, alpha=50, inner=5, seed=0):
    tr, va = X_tr.copy(), X_va.copy()
    skf = StratifiedKFold(inner, shuffle=True, random_state=seed)
    for name, spec in SEGS.items():
        k_tr, k_va = _key(X_tr, spec), _key(X_va, spec)
        oof = np.zeros(len(X_tr))
        for a, b in skf.split(X_tr, y_tr):
            oof[b] = _enc(k_tr.iloc[a], y_tr.iloc[a], k_tr.iloc[b], alpha)
        tr[name] = oof
        va[name] = _enc(k_tr, y_tr, k_va, alpha)
    return tr, va


seg = []
for tr, va in SPLITS:
    Xtr, Xva = add_segment_risk(X_fe.iloc[tr], y.iloc[tr], X_fe.iloc[va])
    m = xgb.XGBClassifier(**XGB).fit(Xtr, y.iloc[tr])
    seg.append(roc_auc_score(y.iloc[va], m.predict_proba(Xva)[:, 1]))
record("+ riesgo por segmento (en el fold)", np.array(seg))

# %% [markdown]
# ## 5. Valor de la limpieza para un modelo lineal

# %%
logreg = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=1.0, max_iter=3000))
lin_cols = ["util_clip", "log_income", "log_debt_amount", "debt_ratio_valid", "age", "total_late",
            "max_late", "any_late", "late_severity", "late_sentinel", "income_missing", "income_zero",
            "dependents_missing", "non_re_lines", "NumberOfDependents",
            "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "util_gt1"]
X_lin = X_fe[lin_cols].copy()
X_lin[["total_late", "max_late", "late_severity"]] = X_lin[["total_late", "max_late", "late_severity"]].clip(upper=10)
lin = pd.DataFrame({"modelo": ["logreg · crudas", "logreg · limpias y transformadas"],
                    "auc": [fold_aucs(train[RAW], logreg).mean(), fold_aucs(X_lin, logreg).mean()]})
lin.round(5)

# %% [markdown]
# ## 6. Ablación hacia atrás: ¿cuál es el set mínimo que no pierde AUC?

# %%
from features.features import clean  # noqa: E402

clean_df = clean(train.drop(columns=[TARGET, ID]))
GROUPS = {
    "agregados de atrasos": ["total_late", "max_late", "any_late", "late_severity"],
    "derivados de ingreso y deuda": ["log_income", "debt_ratio_valid", "debt_amount",
                                     "log_debt_amount", "income_per_dep"],
    "utilización derivada": ["util_clip", "util_gt1"],
    "interacciones y estructura": ["non_re_lines", "age_x_util", "late_x_util"],
}
compare("solo crudas (sin limpieza)", train[RAW])
compare("crudas limpias + banderas", clean_df)
compare("mínimo: limpias + total_late + any_late",
        pd.concat([clean_df, X_fe[["total_late", "any_late"]]], axis=1))
for g, cols in GROUPS.items():
    out = compare(f"base sin {g}", X_fe.drop(columns=cols))
out

# %% [markdown]
# ## 7. Regresión logística: qué features a mano ayudan a un modelo lineal

# %%
extra_lin = pd.concat([X_lin, spikes[["util_eq0", "util_095_1"]],
                       exposure[["has_90", "only_mild_late"]]], axis=1)
lin2 = pd.DataFrame({
    "modelo": ["logreg · limpias y transformadas", "+ util_eq0, util_095_1, has_90, only_mild_late"],
    "auc": [fold_aucs(X_lin, logreg).mean(), fold_aucs(extra_lin, logreg).mean()]})
lin2.round(5)

# %% [markdown]
# ## 8. ¿Hay interacciones? Profundidad del árbol como diagnóstico
# Si max_depth=1 (modelo aditivo) empata con max_depth=4, no hay interacciones que explotar.
# Se escala n_estimators para que los árboles someros no queden subajustados (solo diagnóstico).

# %%
N_EST = {1: 1200, 2: 600, 3: 400, 4: 300, 6: 200}
rows = []
for depth, n in N_EST.items():
    a = fold_aucs(X_fe, lambda d=depth, n=n: xgb.XGBClassifier(**{**XGB, "max_depth": d, "n_estimators": n}))
    rows.append((depth, n, a.mean(), (a - BASE).mean()))
pd.DataFrame(rows, columns=["max_depth", "n_estimators", "auc", "delta_vs_base"]).round(5)

# %% [markdown]
# ## 9. Análisis de errores (predicciones fuera de muestra, 5 folds disjuntos)

# %%
oof = np.zeros(len(y))
for tr, va in SPLITS[:5]:
    m = xgb.XGBClassifier(**XGB).fit(X_fe.iloc[tr], y.iloc[tr])
    oof[va] = m.predict_proba(X_fe.iloc[va])[:, 1]
print("AUC OOF:", round(roc_auc_score(y, oof), 5))

df = pd.DataFrame({"y": y, "p": oof, "age": X_fe["age"], "util": X_fe["util_clip"],
                   "any_late": X_fe["any_late"], "income_missing": X_fe["income_missing"]})


def slice_report(by):
    g = df.groupby(by, observed=True)
    return pd.DataFrame({
        "n": g.size(), "tasa_real": g["y"].mean(), "pred_media": g["p"].mean(),
        "auc": g.apply(lambda s: roc_auc_score(s["y"], s["p"]) if s["y"].nunique() == 2 else np.nan),
    }).round(4)


display(slice_report(pd.cut(df["age"], [17, 30, 40, 50, 60, 70, 110])))
display(slice_report(pd.cut(df["util"], [-0.01, 0, 0.05, 0.3, 0.6, 1, 2])))
display(slice_report("any_late"))
display(slice_report("income_missing"))

# %%
fn = df[(df["y"] == 1) & (df["p"] < 0.02)]
print(f"Positivos con p < 0.02: {len(fn)} de {int(df['y'].sum())} ({len(fn) / df['y'].sum():.1%})")
cols = ["age", "util_clip", "total_late", "MonthlyIncome", "DebtRatio", "NumberOfOpenCreditLinesAndLoans"]
display(X_fe.loc[fn.index, cols].describe().round(2))
display(X_fe.loc[(df["y"] == 0), cols].describe().round(2))

# %% [markdown]
# ## 10. Confirmación en folds nuevos (semilla 7, 5×3) de los sets candidatos

# %%
SPLITS_NEW = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7).split(X_fe, y))


def fold_aucs_new(X, factory=xgb_model):
    return np.array([
        roc_auc_score(y.iloc[va], factory().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1])
        for tr, va in SPLITS_NEW
    ])


SETS = {
    "mínimo (limpias + total_late + any_late)": pd.concat([clean_df, X_fe[["total_late", "any_late"]]], axis=1),
    "base completa": X_fe,
    "base sin utilización derivada": X_fe.drop(columns=GROUPS["utilización derivada"]),
}
fresh = {k: fold_aucs_new(v) for k, v in SETS.items()}
ref = fresh["base completa"]
pd.DataFrame({k: dict(n_cols=SETS[k].shape[1], auc=a.mean(), delta_vs_base=(a - ref).mean(),
                      se=(a - ref).std(ddof=1) / np.sqrt(len(a))) for k, a in fresh.items()}).T.round(5)

# %% [markdown]
# ### Ingreso faltante y centinelas: ¿explican el AUC alto del grupo?

# %%
sent = X_fe["late_sentinel"] == 1
print(pd.crosstab(X_fe["income_missing"], sent, margins=True))
for im in (0, 1):
    mk = (df["income_missing"] == im) & ~sent
    print(f"income_missing={im}, sin centinelas: n={mk.sum()}, AUC={roc_auc_score(df.loc[mk, 'y'], df.loc[mk, 'p']):.4f}")

# %% [markdown]
# ## 11. Decisiones
# - Descartadas (ΔAUC dentro de ±2 SE): cohorte por edad, picos y redondez, exposición y composición,
#   patrón de faltantes, frecuencia de ingreso, riesgo por segmento en el fold.
# - Interacciones (sección 8): el modelo aditivo pierde 0.0025 (profundidad 1: 0.8627 vs profundidad 3-4: 0.8652).
#   Las interacciones aportan como máximo ese 0.0025, ya capturado con profundidad 3 o 4; profundidad 6 empeora.
# - Calibración (sección 9): la predicción media coincide con la tasa real en todos los cortes.
# - Subgrupos: AUC 0.831 (17-30) y 0.841 (30-40) frente a 0.854-0.856 de 40 años en adelante.
#   Tasa de impago más alta en jóvenes (11.9%). Reportar en el análisis de equidad.
# - Errores irreducibles: 520 de 7,018 positivos (7.4%) reciben p < 0.02 y su perfil se parece al de los
#   no deudores típicos (utilización mediana 0.06, sin atrasos).
# - Ingreso faltante: AUC 0.901 vs 0.855 con ingreso; ver la celda de centinelas.
# - Folds nuevos (semilla 7, 5x3): mínimo (16 columnas) = base completa (28) = 0.86529. El +0.00024 de
#   quitar la utilización derivada no se replicó (efecto de elegir el mejor entre varios).
# - SET CANDIDATO FINAL: mínimo de 16 columnas (`prepare_min`): originales limpias, 4 banderas,
#   total_late y any_late.
