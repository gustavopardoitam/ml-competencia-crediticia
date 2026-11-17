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
# # 04 · Balanceo de clases, tuning y ensamble (set mínimo de 16 columnas)
# Protocolo: comparaciones con folds 5x2 (semilla 42); tuning con 5x1 (semilla 123);
# confirmación final y ensamble con 5x3 (semilla 2024). Ninguna decisión se evalúa donde se tomó.

# %%
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from features.features import ID, TARGET, UTIL, prepare, prepare_min  # noqa: E402

train = pd.read_csv(ROOT / "data/raw/train.csv")
test = pd.read_csv(ROOT / "data/raw/test.csv")
y = train[TARGET].astype(int)
X = prepare_min(train)
print(X.shape, "| tasa positiva:", round(y.mean(), 4))

XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
           random_state=42, n_jobs=-1)

# %% [markdown]
# ## 1. Balanceo de clases (dentro de cada fold; la validación nunca se toca)
# Se compara contra XGBoost por defecto. `pred_media` frente a la tasa real (0.0668) mide la calibración.

# %%
SPLITS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y))
SPW = (y == 0).sum() / (y == 1).sum()


def evaluate(fit_predict, splits):
    aucs, means = [], []
    for tr, va in splits:
        p = fit_predict(tr, va)
        aucs.append(roc_auc_score(y.iloc[va], p))
        means.append(p.mean())
    return np.array(aucs), float(np.mean(means))


def fp_xgb(**kw):
    def f(tr, va):
        m = xgb.XGBClassifier(**{**XGB, **kw}).fit(X.iloc[tr], y.iloc[tr])
        return m.predict_proba(X.iloc[va])[:, 1]
    return f


def fp_under(ratio, seed=0):
    def f(tr, va):
        rng = np.random.RandomState(seed)
        pos, neg = tr[y.iloc[tr].values == 1], tr[y.iloc[tr].values == 0]
        neg = rng.choice(neg, size=min(len(neg), int(ratio * len(pos))), replace=False)
        idx = np.concatenate([pos, neg])
        m = xgb.XGBClassifier(**XGB).fit(X.iloc[idx], y.iloc[idx])
        return m.predict_proba(X.iloc[va])[:, 1]
    return f


def fp_smote(strategy=0.25):
    # Nota: SMOTE exige imputar; esta variante pierde el manejo nativo de NaN de XGBoost.
    def f(tr, va):
        pipe = make_imb_pipeline(SimpleImputer(strategy="median"),
                                 SMOTE(sampling_strategy=strategy, random_state=42),
                                 xgb.XGBClassifier(**XGB))
        pipe.fit(X.iloc[tr], y.iloc[tr])
        return pipe.predict_proba(X.iloc[va])[:, 1]
    return f


rows, BASE = [], None


def record(name, fp):
    global BASE
    a, pm = evaluate(fp, SPLITS)
    if BASE is None:
        BASE = a
    d = a - BASE
    rows.append(dict(variante=name, auc=a.mean(), delta=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                     folds_mejores=int((d > 0).sum()), pred_media=pm))
    return pd.DataFrame(rows).round(5)


record("base: XGBoost por defecto", fp_xgb())
record("scale_pos_weight = 3", fp_xgb(scale_pos_weight=3))
record(f"scale_pos_weight = {SPW:.1f} (neg/pos)", fp_xgb(scale_pos_weight=SPW))
record("submuestreo 3:1 (neg:pos)", fp_under(3))
record("submuestreo 1:1", fp_under(1))
record("SMOTE 0.25 (imputación mediana)", fp_smote(0.25))

# %% [markdown]
# ## 2. Búsqueda de hiperparámetros con Optuna (5x1, semilla 123, early stopping dentro del fold)
# Espacio: profundidad 2 a 5 (el diagnóstico de profundidad mostró que 3 o 4 es lo óptimo).
# No se incluye scale_pos_weight: se decide con la sección 1.

# %%
optuna.logging.set_verbosity(optuna.logging.WARNING)
TUNE_SPLITS = list(StratifiedKFold(5, shuffle=True, random_state=123).split(X, y))


def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 5),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        gamma=trial.suggest_float("gamma", 0, 5),
    )
    scores, iters = [], []
    for i, (tr, va) in enumerate(TUNE_SPLITS):
        m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, tree_method="hist",
                              eval_metric="auc", random_state=42, n_jobs=-1, **params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        scores.append(m.best_score)
        iters.append(m.best_iteration + 1)
        trial.report(float(np.mean(scores)), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("n_estimators", int(np.median(iters) * 1.1))
    return float(np.mean(scores))


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
study.enqueue_trial(dict(learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=1e-3, gamma=0.0))
study.optimize(objective, n_trials=40, timeout=2400, show_progress_bar=True)

# %%
best = study.best_trial
print("AUC del punto de partida (parámetros por defecto):", round(study.trials[0].value, 5))
print("Mejor AUC en tuning (ligeramente optimista por el early stopping):", round(best.value, 5))
print("n_estimators sugerido:", best.user_attrs["n_estimators"])
display(pd.Series(best.params).round(4))
tdf = study.trials_dataframe()
display(tdf.sort_values("value", ascending=False).head(10)[
    [c for c in tdf.columns if c.startswith("params_")] + ["value"]].round(4))
(ROOT / "src/config").mkdir(parents=True, exist_ok=True)
json.dump({**best.params, "n_estimators": best.user_attrs["n_estimators"], "cv_auc_tuning": best.value},
          open(ROOT / "src/config/xgb_best_params.json", "w"), indent=2)

# %% [markdown]
# ## 3. Confirmación en folds nuevos (5x3, semilla 2024) y ensamble
# XGBoost por defecto vs. ajustado (sin early stopping), LightGBM y regresión logística limpia.
# Los pesos de las mezclas se fijan a priori para no inflar el resultado.

# %%
lin_cols = ["util_clip", "log_income", "log_debt_amount", "debt_ratio_valid", "age", "total_late",
            "max_late", "any_late", "late_severity", "late_sentinel", "income_missing", "income_zero",
            "dependents_missing", "non_re_lines", "NumberOfDependents",
            "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "util_gt1"]


def build_lin(df):
    F = prepare(df)
    L = F[lin_cols].copy()
    L[["total_late", "max_late", "late_severity"]] = L[["total_late", "max_late", "late_severity"]].clip(upper=10)
    L["util_eq0"] = (F[UTIL] == 0).astype(int)
    L["has_90"] = (F["NumberOfTimes90DaysLate"].fillna(0) > 0).astype(int)
    return L


X_lin = build_lin(train)
TUNED = {**XGB, **study.best_params, "n_estimators": study.best_trial.user_attrs["n_estimators"]}
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=-1)
FACTORIES = {
    "xgb_default": (lambda: xgb.XGBClassifier(**XGB), X),
    "xgb_tuned": (lambda: xgb.XGBClassifier(**TUNED), X),
    "lgbm": (lambda: lgb.LGBMClassifier(**LGB), X),
    "logreg": (lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     LogisticRegression(C=1.0, max_iter=3000)), X_lin),
}
SPLITS_FINAL = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=2024).split(X, y))
preds = {k: [] for k in FACTORIES}
for tr, va in SPLITS_FINAL:
    for k, (f, XX) in FACTORIES.items():
        preds[k].append(f().fit(XX.iloc[tr], y.iloc[tr]).predict_proba(XX.iloc[va])[:, 1])
yva = [y.iloc[va].values for _, va in SPLITS_FINAL]


def fold_auc(p_list):
    return np.array([roc_auc_score(a, b) for a, b in zip(yva, p_list)])


def blend(names, weights=None, rank=False):
    w = weights or [1 / len(names)] * len(names)
    out = []
    for i in range(len(SPLITS_FINAL)):
        parts = [rankdata(preds[n][i]) / len(preds[n][i]) if rank else preds[n][i] for n in names]
        out.append(sum(wi * pi for wi, pi in zip(w, parts)))
    return out


cands = dict(preds)
cands["mezcla xgb_tuned + lgbm"] = blend(["xgb_tuned", "lgbm"])
cands["mezcla xgb_tuned (0.8) + logreg (0.2)"] = blend(["xgb_tuned", "logreg"], [0.8, 0.2])
cands["mezcla xgb_tuned (0.5) + lgbm (0.3) + logreg (0.2)"] = blend(["xgb_tuned", "lgbm", "logreg"], [0.5, 0.3, 0.2])
cands["mezcla por rangos, pesos iguales (3 modelos)"] = blend(["xgb_tuned", "lgbm", "logreg"], rank=True)
ref = fold_auc(preds["xgb_default"])
res = []
for k, pl in cands.items():
    a = fold_auc(pl)
    d = a - ref
    res.append(dict(modelo=k, auc=a.mean(), delta_vs_xgb_default=d.mean(),
                    se=d.std(ddof=1) / np.sqrt(len(d)), folds_mejores=int((d > 0).sum())))
display(pd.DataFrame(res).round(5))
# Diversidad entre modelos (Spearman de las predicciones de la primera repetición)
pd.DataFrame({k: np.concatenate(preds[k][:5]) for k in FACTORIES}).corr(method="spearman").round(3)

# %% [markdown]
# ## 4. Submissions (se generan; se sube a Kaggle a mano y se anota el AUC del leaderboard)

# %%
Xt = prepare_min(test)[X.columns]
Xt_lin = build_lin(test)[X_lin.columns]
full = {k: f().fit(XX, y) for k, (f, XX) in FACTORIES.items() if k != "xgb_default"}
pt = {"xgb_tuned": full["xgb_tuned"].predict_proba(Xt)[:, 1],
      "lgbm": full["lgbm"].predict_proba(Xt)[:, 1],
      "logreg": full["logreg"].predict_proba(Xt_lin)[:, 1]}
sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
assert set(test[ID]) == set(sample[sample.columns[0]])


def write_sub(name, p):
    sub = pd.DataFrame({sample.columns[0]: test[ID], sample.columns[1]: p})
    assert len(sub) == len(sample) and sub.iloc[:, 1].between(0, 1).all()
    (ROOT / "submissions").mkdir(exist_ok=True)
    sub.to_csv(ROOT / f"submissions/{name}.csv", index=False)
    print("escrito:", f"submissions/{name}.csv", sub.shape)


write_sub("xgb_tuned", pt["xgb_tuned"])
write_sub("blend_3", 0.5 * pt["xgb_tuned"] + 0.3 * pt["lgbm"] + 0.2 * pt["logreg"])

# %% [markdown]
# ## 5. Decisiones (completar)
# - Balanceo de clases (sección 1):
# - Hiperparámetros elegidos (sección 2) y mejora real en folds nuevos (sección 3):
# - Ensamble: qué mezcla se queda y si la mejora supera ~2 SE:
# - AUC en Kaggle (public leaderboard) vs. CV de 5x3:
