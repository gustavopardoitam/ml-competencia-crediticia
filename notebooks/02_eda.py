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
# # 02 · EDA básico — Competencia crediticia
#
# Versión **paso a paso** del análisis exploratorio. Cada sección responde una
# pregunta sencilla y explica qué se está viendo. Para un análisis más
# avanzado consulta `01_eda.ipynb`.
#
# **Objetivo del modelo:** predecir la probabilidad de que un cliente tenga
# morosidad grave (90 días o más de atraso) en los próximos 2 años.
# **Métrica:** AUC-ROC.
#
# **Qué se podrá experimentar en la competencia** (y dónde ayuda este EDA):
#
# | Tema | Qué nos dice este EDA |
# | --- | --- |
# | Calidad de datos (duplicados) | Sección 1 |
# | Valores faltantes y atípicos | Secciones 5 y 7 |
# | Ingeniería y selección de variables | Secciones 8 y 9 |
# | Algoritmos de clasificación | Sección 10 (desbalance y escalas) |
# | Regularización, validación cruzada, hiperparámetros | Sección 10 |

# %% [markdown]
# ## 0. Preparación
# Cargamos las librerías y los datos. Los archivos viven en `data/raw/`
# (se descargan con `PYTHONPATH=src uv run python -m etl.download`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
train = pd.read_csv(ROOT / "data/raw/train.csv")
test = pd.read_csv(ROOT / "data/raw/test.csv")

TARGET = "SeriousDlqin2yrs"
ID = "Id"
FEATS = [c for c in train.columns if c not in (TARGET, ID)]

# %% [markdown]
# ## 1. Ordenamiento y registros duplicados
# Antes de explorar, revisamos la **calidad básica** del dataset: ¿hay
# registros repetidos?
#
# **Criterio de eliminación:** solo se elimina una fila si su `Id` está
# repetido (el mismo registro cargado dos veces). Las filas con `Id` distinto se
# conservan aunque su contenido sea idéntico, porque pueden ser clientes
# distintos con el mismo perfil; esas solo se **reportan**.
#
# Primero ordenamos los datos:
# - Por `Id`, para tener un orden estable y reproducible.
# - Por todas las variables, para que los registros iguales queden juntos y se
#   puedan revisar a simple vista.

# %%
train = train.sort_values(ID).reset_index(drop=True)
test = test.sort_values(ID).reset_index(drop=True)

# Orden por todas las columnas: los registros idénticos quedan contiguos
train_ordenado = train.sort_values(FEATS + [TARGET], na_position="last")
train_ordenado.head()

# %% [markdown]
# Revisamos tres tipos de duplicado:
#
# | Tipo | Definición | Acción |
# | --- | --- | --- |
# | Por `Id` | El mismo `Id` aparece más de una vez | **Se elimina** la copia |
# | Mismo contenido | Mismas variables y mismo objetivo, distinto `Id` | Se reporta, se conserva |
# | En conflicto | Mismas variables pero objetivo distinto | Se reporta, se conserva |

# %%
dup_id = train[ID].duplicated()
dup_exacto = train.duplicated(subset=FEATS + [TARGET], keep="first")
dup_variables = train.duplicated(subset=FEATS, keep=False)
grupos_conflicto = (
    train[dup_variables]
    .groupby(FEATS, dropna=False)[TARGET]
    .nunique()
    .gt(1)
    .sum()
)

resumen_dup = pd.DataFrame(
    {
        "registros": [
            dup_id.sum(),
            dup_exacto.sum(),
            train.duplicated(subset=FEATS + [TARGET], keep=False).sum(),
            grupos_conflicto,
        ]
    },
    index=[
        "Id repetidos",
        "Mismo contenido con distinto Id (copias)",
        "Filas involucradas en contenido repetido",
        "Grupos en conflicto (mismo perfil, objetivo distinto)",
    ],
)
resumen_dup

# %% [markdown]
# ### Notificación

# %%
n_dup_id = int(dup_id.sum())
n_dup_contenido = int(dup_exacto.sum())
if n_dup_id > 0:
    print(f"⚠️ Hay {n_dup_id:,} Id repetidos: se eliminarán las copias.")
else:
    print("✅ No hay Id repetidos: no se elimina ninguna fila.")

if n_dup_contenido > 0:
    pct = n_dup_contenido / len(train) * 100
    print(
        f"ℹ️ Hay {n_dup_contenido:,} registros con el mismo contenido y distinto Id "
        f"({pct:.2f} % de train). Se conservan porque su Id no se repite."
    )

if grupos_conflicto > 0:
    print(
        f"ℹ️ {grupos_conflicto} grupos tienen el mismo perfil con objetivo distinto. "
        "No se eliminan (ambos resultados son reales), pero son ruido para el modelo."
    )

# %%
# Ejemplos: registros duplicados vistos juntos gracias al ordenamiento
mascara = train_ordenado.duplicated(subset=FEATS + [TARGET], keep=False)
train_ordenado[mascara].head(10)

# %%
# ¿Cómo son los duplicados frente al total?
filas_dup = train.duplicated(subset=FEATS + [TARGET], keep=False)
pd.DataFrame(
    {
        "duplicados": [
            train.loc[filas_dup, TARGET].mean(),
            train.loc[filas_dup, "MonthlyIncome"].isna().mean(),
            train.loc[filas_dup, "age"].median(),
        ],
        "todo train": [
            train[TARGET].mean(),
            train["MonthlyIncome"].isna().mean(),
            train["age"].median(),
        ],
    },
    index=["tasa de morosidad", "proporción sin ingreso", "edad mediana"],
).round(3)

# %% [markdown]
# **Lectura:** los duplicados son, en su mayoría, perfiles "delgados": sin
# ingreso reportado y con casi todas las variables en cero. Es posible que
# algunos sean personas distintas con el mismo perfil, pero como tienen
# exactamente la misma información y el mismo resultado, en el modelado pesan
# doble. Como su `Id` es distinto, **se conservan**. Si más adelante se quiere
# probar su efecto, se puede comparar el AUC con y sin ellos en la validación
# cruzada.

# %% [markdown]
# ### Limpieza y cifras de control
# Eliminamos solo las filas con `Id` repetido (se conserva la primera) y
# verificamos con cifras de control que se quitó exactamente lo esperado.

# %%
filas_antes = len(train)
tasa_antes = train[TARGET].mean()

train = train.drop_duplicates(subset=ID, keep="first").reset_index(drop=True)

filas_despues = len(train)
tasa_despues = train[TARGET].mean()
cifras_control = pd.DataFrame(
    {
        "filas": [filas_antes, filas_despues, filas_despues - filas_antes],
        "tasa de morosidad": [tasa_antes, tasa_despues, tasa_despues - tasa_antes],
    },
    index=["antes", "después", "diferencia"],
)
display(cifras_control.style.format({"tasa de morosidad": "{:.4f}"}))

# Reglas de calidad (DQ): si alguna falla, el notebook se detiene
assert filas_antes - filas_despues == n_dup_id, "Se eliminaron filas de más o de menos"
assert train[ID].is_unique, "Quedan Id repetidos"
print(f"✅ Dataset listo: {filas_despues:,} filas ({filas_antes - filas_despues} eliminadas).")

# %% [markdown]
# ### ¿Y en `test`?

# %%
n_dup_test = int(test.duplicated(subset=FEATS).sum())
print(f"Registros de test con perfil repetido: {n_dup_test}")
print("ℹ️ En test NO se elimina nada: la competencia pide una predicción por cada Id.")

# %% [markdown]
# **A partir de aquí, todo el análisis usa `train` sin `Id` repetidos.**

# %% [markdown]
# ## 2. Diccionario de variables
# Qué significa cada columna. Es la base para interpretar todo lo demás.

# %%
DICCIONARIO = {
    "SeriousDlqin2yrs": "OBJETIVO. 1 = morosidad grave (90+ días) en 2 años; 0 = no.",
    "RevolvingUtilizationOfUnsecuredLines": "Saldo en tarjetas y líneas de crédito / límite total. Debería estar entre 0 y 1.",
    "age": "Edad del cliente en años.",
    "NumberOfTime30-59DaysPastDueNotWorse": "Veces con atraso de 30 a 59 días (sin llegar a peor).",
    "DebtRatio": "Pagos mensuales de deuda / ingreso mensual bruto.",
    "MonthlyIncome": "Ingreso mensual.",
    "NumberOfOpenCreditLinesAndLoans": "Número de créditos y líneas abiertas.",
    "NumberOfTimes90DaysLate": "Veces con atraso de 90 días o más.",
    "NumberRealEstateLoansOrLines": "Número de créditos hipotecarios o inmobiliarios.",
    "NumberOfTime60-89DaysPastDueNotWorse": "Veces con atraso de 60 a 89 días (sin llegar a peor).",
    "NumberOfDependents": "Número de dependientes económicos (sin contar al cliente).",
}
pd.Series(DICCIONARIO, name="descripción").to_frame()

# %% [markdown]
# ## 3. Tamaño y primeras filas
# ¿Cuántas filas y columnas hay? ¿Cómo se ve un registro?

# %%
print(f"train: {train.shape[0]:,} filas x {train.shape[1]} columnas")
print(f"test : {test.shape[0]:,} filas x {test.shape[1]} columnas")
print("Columnas que están en train pero no en test:", set(train.columns) - set(test.columns))
train.head()

# %% [markdown]
# **Lectura:** `test` no trae la columna objetivo; es lo que el modelo debe
# predecir. Solo podemos medir qué tan bien funciona con validación sobre
# `train`.

# %% [markdown]
# ## 4. Variable objetivo
# ¿Qué proporción de clientes cae en morosidad grave?

# %%
conteo = train[TARGET].value_counts().rename("clientes").to_frame()
conteo["porcentaje"] = (train[TARGET].value_counts(normalize=True) * 100).round(2)
display(conteo)

ax = sns.countplot(data=train, x=TARGET)
ax.set_title("Distribución de la variable objetivo")
ax.set_xlabel("0 = no moroso · 1 = moroso grave")
plt.show()

# %% [markdown]
# **Lectura:** la clase `1` es muy minoritaria (**datos desbalanceados**).
# Consecuencias para el modelado:
# - Accuracy no sirve: predecir siempre `0` ya daría ~93 %. Por eso la métrica es AUC-ROC.
# - Conviene usar **validación cruzada estratificada** (mantiene la proporción en cada partición).
# - Algunos modelos aceptan pesos de clase (`class_weight`).

# %% [markdown]
# ## 5. Valores faltantes
# ¿Qué columnas tienen datos vacíos y cuántos?

# %%
faltantes = pd.DataFrame(
    {
        "n_faltantes": train[FEATS].isna().sum(),
        "% train": (train[FEATS].isna().mean() * 100).round(2),
        "% test": (test[FEATS].isna().mean() * 100).round(2),
    }
)
faltantes[faltantes["n_faltantes"] > 0]

# %%
# ¿Quien no reporta ingreso es más o menos moroso?
(
    train.groupby(train["MonthlyIncome"].isna())[TARGET]
    .agg(clientes="count", tasa_morosidad="mean")
    .rename_axis("ingreso_faltante")
    .round(4)
)

# %% [markdown]
# **Lectura:** faltan datos en `MonthlyIncome` y en `NumberOfDependents`. El
# patrón de faltantes es parecido en `train` y `test`.
# Si la tasa de morosidad cambia cuando falta el ingreso, el hecho de que falte
# **ya es información**: una opción es crear una columna indicadora
# (`ingreso_faltante`) además de imputar (por ejemplo con la mediana).

# %% [markdown]
# ## 6. Estadísticos descriptivos
# Mínimo, máximo, media y percentiles de cada variable numérica.

# %%
train[FEATS].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(2)

# %% [markdown]
# **Cómo leer la tabla:** compara `50%` (mediana) con `mean` y con `max`.
# - Si `max` es enorme frente al `99%`, hay **valores atípicos** (outliers).
# - Si `mean` está muy lejos de la mediana, la distribución tiene cola larga.
#
# Cosas que suelen llamar la atención aquí: `age` con mínimo 0,
# `RevolvingUtilizationOfUnsecuredLines` muy por encima de 1 y las variables de
# atraso con valores 96 o 98 (códigos especiales, no conteos reales).

# %% [markdown]
# ## 7. Valores atípicos
# Revisamos tres casos concretos.

# %%
print("Edad menor a 18:", (train["age"] < 18).sum())
print("Edad mayor a 90:", (train["age"] > 90).sum())
print("Utilización de líneas > 1:", (train["RevolvingUtilizationOfUnsecuredLines"] > 1).sum())
print("DebtRatio > 10:", (train["DebtRatio"] > 10).sum())

LATE = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
print("\nValores más altos en variables de atraso:")
for c in LATE:
    print(f"  {c}: {sorted(train[c].unique())[-4:]}")

# %%
# Los valores 96 y 98 parecen códigos especiales: ¿cómo se comportan?
codigo = (train[LATE] >= 96).any(axis=1)
print("Filas con código 96/98:", codigo.sum())
print("Tasa de morosidad CON código   :", round(train.loc[codigo, TARGET].mean(), 4))
print("Tasa de morosidad SIN código   :", round(train.loc[~codigo, TARGET].mean(), 4))

# %% [markdown]
# **Lectura:** los 96/98 no son 96 atrasos reales: son marcas especiales y
# además están asociadas a una tasa de morosidad muy distinta. Opciones:
# tratarlos como categoría aparte, ponerlos en `NaN` y crear un indicador, o
# recortar (*winsorizar*) el valor. Edades de 0 y razones de utilización o
# deuda extremas se pueden recortar a un percentil alto (p. ej. 99).

# %% [markdown]
# ## 8. Distribución de cada variable
# Histogramas recortados al percentil 99 para que las colas extremas no
# aplasten el gráfico.

# %%
fig, axes = plt.subplots(3, 4, figsize=(18, 10))
for ax, col in zip(axes.ravel(), FEATS):
    serie = train[col].dropna()
    serie = serie[serie <= serie.quantile(0.99)]
    ax.hist(serie, bins=30, color="steelblue")
    ax.set_title(col, fontsize=9)
for ax in axes.ravel()[len(FEATS):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Lectura:** la mayoría de las variables son muy asimétricas (muchos valores
# pequeños y pocos grandes). Para modelos lineales ayuda aplicar `log1p` o
# discretizar; los modelos de árboles no lo necesitan.

# %% [markdown]
# ## 9. Relación de cada variable con el objetivo
# ¿Qué variables parecen separar a morosos de no morosos?

# %%
corr = train[FEATS + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
corr.sort_values().plot.barh(figsize=(8, 5), color="steelblue")
plt.title("Correlación de Spearman con SeriousDlqin2yrs")
plt.show()

# %%
# Tasa de morosidad por rangos de edad (cuantiles)
rangos = pd.qcut(train["age"], 8)
train.groupby(rangos, observed=True)[TARGET].mean().plot(kind="bar", figsize=(8, 4))
plt.title("Tasa de morosidad por rango de edad")
plt.ylabel("tasa de morosidad")
plt.show()

# %%
# Tasa de morosidad según número de atrasos de 90+ días (0 a 5)
t = train[train["NumberOfTimes90DaysLate"] <= 5]
t.groupby("NumberOfTimes90DaysLate")[TARGET].mean().plot(kind="bar", figsize=(8, 4))
plt.title("Tasa de morosidad según atrasos de 90+ días")
plt.ylabel("tasa de morosidad")
plt.show()

# %% [markdown]
# **Lectura:** las variables de atraso previo son las más relacionadas con la
# morosidad futura, seguidas de la utilización de líneas de crédito
# (`RevolvingUtilizationOfUnsecuredLines`). La edad tiene relación inversa (a
# mayor edad, menor riesgo). Ideas de **ingeniería de variables** que salen de aquí:
# - Suma total de atrasos (30-59 + 60-89 + 90+).
# - Indicador de "tuvo algún atraso".
# - Ingreso por dependiente (`MonthlyIncome / (NumberOfDependents + 1)`).
# - Indicadores de faltante y de códigos 96/98.

# %% [markdown]
# ## 10. Resumen y siguientes pasos
#
# | Hallazgo | Qué hacer después |
# | --- | --- |
# | Sin `Id` repetidos; 339 filas con mismo contenido y distinto `Id` | No se elimina nada; probar su efecto en validación cruzada |
# | Perfiles idénticos con objetivo distinto | Se conservan; son ruido irreducible |
# | Clase positiva muy minoritaria | Validación cruzada estratificada, AUC-ROC, `class_weight` |
# | Faltantes en ingreso y dependientes | Imputar (mediana) + columna indicadora |
# | Códigos 96/98 en variables de atraso | Tratarlos aparte (NaN + indicador) |
# | Edad 0, utilización y DebtRatio extremos | Recortar a percentil alto o limpiar |
# | Variables muy asimétricas | `log1p` para modelos lineales; árboles no lo requieren |
# | Atrasos previos son los más predictivos | Crear variables agregadas de atrasos |
#
# **Plan de experimentos sugerido:**
# 1. Línea base: regresión logística con regularización (L1/L2) sobre datos limpios y escalados.
# 2. Modelos de árboles: Random Forest y Gradient Boosting (manejan asimetría y no linealidad).
# 3. Validación cruzada estratificada de 5 particiones, medida en AUC-ROC.
# 4. Selección de hiperparámetros con búsqueda (`GridSearchCV` / `RandomizedSearchCV`).
# 5. Comparar el efecto de cada decisión de limpieza y de variables nuevas.
#
# **Importante:** las transformaciones (medianas, escalado) se ajustan solo con
# los datos de entrenamiento de cada partición, para evitar fuga de datos.
