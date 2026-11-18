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
# # 06 · Comparación cara a cara: entrega B (develop) vs. línea FE-mínimo
# Se cruzan variables y hiperparámetros para saber qué causa cada diferencia.
# Protocolos: (a) folds del equipo (5, semilla 985); (b) 5x3 con semilla nueva (31415).
# Árboles fijos, sin early stopping, una semilla por modelo.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from evaluation import RANDOM_STATE, get_cv  # noqa: E402
from features.engineering_jae import FeatureConfig, build_preprocessor, prepare_fixed  # noqa: E402
from features.features import ID, TARGET, prepare, prepare_min  # noqa: E402

train = pd.read_csv(ROOT / "data/raw/train.csv")
y = train[TARGET].astype(int)
X_jae = prepare_fixed(train.drop(columns=[TARGET, ID]))
X_fe = prepare(train)
X_min = prepare_min(train)
print(X_jae.shape, X_fe.shape, X_min.shape)

# %%
CFG_ARBOLES = FeatureConfig(for_trees=True, impute_for_trees=False)
BASE = dict(eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
EQUIPO = dict(max_depth=6, min_child_weight=10, subsample=0.6, colsample_bytree=0.7,
              reg_lambda=10.0, gamma=0.5, learning_rate=0.01)
GUS = json.load(open(ROOT / "src/config/xgb_best_params.json"))
GUS.pop("cv_auc_tuning", None)
GUS["max_depth"] = int(GUS["max_depth"])
GUS["n_estimators"] = int(GUS["n_estimators"])
print("params Gustavo:", GUS)


def xgb(**kw):
    return XGBClassifier(**{**BASE, **kw})


MODELS = {
    "jae23_eq": (lambda: Pipeline([("prep", build_preprocessor(CFG_ARBOLES)),
                                   ("clf", xgb(**EQUIPO, n_estimators=764))]), X_jae),
    "fe28_eq": (lambda: xgb(**EQUIPO, n_estimators=777), X_fe),
    "fe28_gus": (lambda: xgb(**GUS), X_fe),
    "min16_eq": (lambda: xgb(**EQUIPO, n_estimators=777), X_min),
    "min16_gus": (lambda: xgb(**GUS), X_min),
}
BLENDS = {
    "B equipo (jae23_eq + fe28_eq)": ["jae23_eq", "fe28_eq"],
    "fe28_eq + min16_gus": ["fe28_eq", "min16_gus"],
    "3 modelos (jae23_eq + fe28_eq + min16_gus)": ["jae23_eq", "fe28_eq", "min16_gus"],
    "4 modelos (+ fe28_gus)": ["jae23_eq", "fe28_eq", "min16_gus", "fe28_gus"],
}
REF = "B equipo (jae23_eq + fe28_eq)"

# %%
def run(splits, etiqueta):
    preds = {k: [] for k in MODELS}
    for tr, va in splits:
        for k, (mk, XX) in MODELS.items():
            preds[k].append(mk().fit(XX.iloc[tr], y.iloc[tr]).predict_proba(XX.iloc[va])[:, 1])
    yva = [y.iloc[va].values for _, va in splits]
    allp = dict(preds)
    for nombre, miembros in BLENDS.items():
        allp[nombre] = [np.mean([preds[m][i] for m in miembros], axis=0) for i in range(len(splits))]
    aucs = {k: np.array([roc_auc_score(a, b) for a, b in zip(yva, pl)]) for k, pl in allp.items()}
    ref = aucs[REF]
    filas = []
    for k, a in aucs.items():
        d = a - ref
        se = d.std(ddof=1) / np.sqrt(len(d)) if k != REF else 0.0
        filas.append(dict(modelo=k, auc=a.mean(), delta_vs_B=d.mean(), se=se,
                          folds_mejores=f"{int((d > 0).sum())}/{len(d)}"))
    print(f"\n=== {etiqueta} ===")
    display(pd.DataFrame(filas).round(5))
    corr = pd.DataFrame({k: np.concatenate(preds[k][:5]) for k in MODELS}).corr(method="spearman")
    display(corr.round(3))
    return aucs


A = run(list(get_cv().split(X_fe, y)), "Protocolo del equipo (5 folds, semilla 985)")

# %%
SPLITS_NEW = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=31415).split(X_fe, y))
B = run(SPLITS_NEW, "Folds nuevos (5x3, semilla 31415)")

# %% [markdown]
# ## Decisiones (completar)
# - ¿Qué pesa más: variables (min16 vs fe28 vs jae23) o hiperparámetros (eq vs gus)?
# - ¿La entrega B sigue ganando en folds nuevos? ¿Alguna combinación la supera por más de ~2 SE?
# - Candidato final para proponer al equipo:
