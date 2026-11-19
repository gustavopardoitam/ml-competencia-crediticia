# ML Championship: competencia crediticia

Predicción de P(SeriousDlqin2yrs = 1). Métrica: AUC-ROC.

## 🥈 Resultado final: 2.º lugar

El equipo **Data Sapiens** obtuvo el **2.º lugar** de la competencia, con
**AUC privado de 0.86509**: subió un lugar respecto a la tabla pública y quedó
a 0.00058 del primero.

**Modelo ganador (entrega F):** ensamble de tres modelos de árboles: dos
XGBoost, uno sobre el conjunto de variables JAE (25 variables) y otro sobre el
conjunto FE (28 variables), y un CatBoost. Los dos XGBoost se promedian 50/50
y el resultado se mezcla por rangos con CatBoost (70 % / 30 %). La ingeniería
de variables limpia los códigos 96/98 de atrasos, lee `DebtRatio` como monto
de deuda cuando falta el ingreso, conserva el ingreso faltante como señal y
agrega interacciones de atrasos × utilización, que son las variables más
usadas por el modelo.

La explicación completa, con el flujo del ensamble y la imputación del ingreso,
está en [`docs/modelo_ganador.md`](docs/modelo_ganador.md).

| Entrega | AUC validación cruzada | AUC público | AUC privado |
| --- | --- | --- | --- |
| XGBoost 2b | 0.86586 | 0.87992 | 0.86411 |
| Entrega B | 0.86663 | 0.88012 | 0.86463 |
| Entrega D | 0.86676 | 0.87996 | 0.86476 |
| **Entrega F (ganadora)** | **0.86698** | **0.88016** | **0.86509** |

## Setup
`uv sync --frozen`. Los datos van en `data/raw/` y no se versionan.

## Reproducir el EDA
1. `git clone` y `uv sync`.
2. Acepta las reglas de la competencia en Kaggle y guarda tu token en `~/.kaggle/access_token` (nunca en el repo).
3. `PYTHONPATH=src uv run python -m etl.download`
4. Abre `notebooks/01_eda.ipynb` con el kernel `.venv`. El `.py` emparejado sirve para revisar diffs; sincroniza con `uv run jupytext --sync notebooks/01_eda.py`.

## Validar un modelo

Todo experimento (combinación de features + modelo) se mide con la misma partición de cross-validation y se registra en una bitácora compartida, para poder comparar resultados de forma justa entre todo el equipo.

### 1. Evaluar

```python
import pandas as pd
from lightgbm import LGBMClassifier
from features.features import prepare  # o elñ conjunto de variables que usaron
from evaluation import evaluate

train = pd.read_csv("data/raw/train.csv")
y = train["SeriousDlqin2yrs"]
X = prepare(train)  # DataFrame numérico, sin Id ni target; los NaN se dejan tal cual

auc_mean, auc_std = evaluate(LGBMClassifier(random_state=985), X, y)
print(f"AUC: {auc_mean:.4f} +/- {auc_std:.4f}")
```

`evaluate(model, X, y)` entrena `model` en 5 folds estratificados (siempre los mismos, ver `RANDOM_STATE` en `src/evaluation/cv.py`) y devuelve `(auc_media, auc_std)`. Acepta cualquier modelo con `.fit`/`.predict_proba`: `LGBMClassifier`, `LogisticRegression`, un `Pipeline` con WOE-encoding, etc.

### 2. Registrar el experimento

```python
from evaluation import log_experiment

log_experiment(
    nombre="baseline_lightgbm",
    features="prepare() - 28 columnas",
    modelo="LGBMClassifier default",
    cv_auc_mean=auc_mean,
    cv_auc_std=auc_std,
    kaggle_score=None,  # se llena al subir el submission a Kaggle
    notas="piso de comparación",
)
```

Cada llamada agrega una fila a `reports/experiments.csv` (se sube al repo para que el equipo vea el historial). Para revisarlo:

```python
from evaluation import read_log
read_log().sort_values("cv_auc_mean", ascending=False)
```

### 3. Sugerencia para comparar experimentos

Cambia **una sola cosa** entre un experimento y el siguiente (o las features, o el modelo, no ambos), para saber qué causó la diferencia. Si la mejora en `cv_auc_mean` es menor que `cv_auc_std`, probablemente es ruido del split y no una mejora real, hace falta algo más grande para confiar en ella.

## EDA: hallazgos, hipótesis y decisiones

Síntesis de dos EDA independientes sobre `data/raw/train.csv`: `notebooks/01_eda.ipynb` y `notebooks/01_eda_karla.ipynb`.

### 1. Calidad de los datos
- Sin filas duplicadas; `Id` único pero no secuencial (rango hasta ~150,000 en 105,000 filas).
- Sin deriva train/test: KS máximo 0.0053, p > 0.47 en todas las variables. El CV local debería reflejar el leaderboard.
- Nulos solo en `MonthlyIncome` (19.8%) y `NumberOfDependents` (2.6%), y **no son aleatorios** (confirmado en ambos EDA): la tasa de impago es menor cuando el dato falta que cuando está presente (p. ej. ingreso faltante 5.6% vs 7.0% con ingreso). Deben tratarse como señal, no solo imputarse.

### 2. Variable objetivo
- Tasa de impago 6.68% (desbalance moderado, ~1 positivo por cada 14 negativos). Validación **estratificada** + **AUC-ROC** como métrica (coinciden ambos EDA).

### 3. Anomalías y códigos centinela
- Las tres columnas de atraso tienen 197 filas con códigos **96/98** (mismas filas en las tres columnas) que no son conteos reales sino datos faltantes codificados como número. Impago en esas filas: 56.9% vs 6.6% en el resto — señal fuerte, pero por ser un código, no un conteo.
- El EDA de `01_eda_karla.ipynb` cuantificó además que **limpiar el centinela apenas cambia el IV** de esas variables (<0.002 de diferencia), porque al binnear quedan agrupados con los conteos altos reales. Esto es tranquilizador para el poder predictivo, pero no exime de limpiarlo: sigue distorsionando cualquier modelo que use la variable en escala continua.
- `age`: 1 fila < 18 años (inválida) y 336 filas > 90 (válidas, revisar). Colas extremas en `RevolvingUtilizationOfUnsecuredLines` (p99 = 1.09, máx 50,708) y `DebtRatio` (p90 = 1,267, máx 329,664).

### 4. Poder predictivo univariado
| Variable | Spearman (`01_eda`) | IV (`01_eda_karla`) |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | 0.24 | 1.12 (muy fuerte) |
| NumberOfTimes90DaysLate | 0.33 | 0.88 (muy fuerte) |
| NumberOfTime30-59DaysPastDueNotWorse | 0.25 | 0.74 (muy fuerte) |
| NumberOfTime60-89DaysPastDueNotWorse | 0.27 | 0.60 (muy fuerte) |
| age | -0.12 | 0.27 (medio) |
| DebtRatio, MonthlyIncome, líneas abiertas, inmuebles, dependientes | ~0 a 0.07 | < 0.1 (débil) |

Ambos EDA coinciden en que las variables de atraso y la utilización dominan el poder predictivo individual.

### 5. Relaciones no lineales
El EDA de `01_eda.ipynb` reportó correlación casi nula para `NumberOfOpenCreditLinesAndLoans` (-0.04) y `NumberRealEstateLoansOrLines` (-0.04). El EDA de `01_eda_karla.ipynb` muestra **por qué**: ambas tienen forma de "U" en la tasa de impago, que una correlación monótona no puede capturar.
- `NumberOfOpenCreditLinesAndLoans`: 0 líneas → ~26% de impago (el riesgo más alto de toda la variable), mínimo ~5% con 6-9 líneas, repunte leve con 20+.
- `NumberRealEstateLoansOrLines`: 0 préstamos → 8.4%, mínimo 5.2-5.6% con 1-2, sube a 6.3%-11.8% con 3-4+.
- Confirmado con información mutua (no asume linealidad): ambas variables suben varias posiciones frente al ranking de Spearman.
- `NumberOfDependents`, en cambio, sí es monótona: 5.9% (0 dependientes) → 9.8% (4+).

### 6. Interacciones y relación ingreso–DebtRatio (EDA `01_eda.ipynb`)
- Interacción edad × utilización: a igual utilización, los jóvenes tienen más riesgo (máx. 42% en 30-40 años con utilización > 1; mín. 18% en mayores de 70).
- `DebtRatio` no es comparable entre filas: con ingreso faltante se comporta como un monto (mediana 1,158); con ingreso presente es una razón (mediana 0.30), salvo en filas con ingreso = 0, donde también se dispara.

### Hipótesis: estado y siguiente paso

| Hipótesis | Estado | Siguiente paso |
|---|---|---|
| Centinela 96/98 = dato faltante, no conteo | **Confirmada** (dos EDA independientes + prueba de sensibilidad de IV) | Limpiar a `NaN` + bandera `late_sentinel` |
| Ausencia de `MonthlyIncome`/`NumberOfDependents` no es aleatoria | **Confirmada** (ambos EDA) | Mantener banderas `*_missing` en el pipeline |
| `NumberOfOpenCreditLinesAndLoans` y `NumberRealEstateLoansOrLines` tienen señal real pero no lineal | **Detectada, por validar en modelo** | Confirmar con importancia de features / SHAP tras entrenar un baseline de árboles; si se usa modelo lineal, discretizar o usar WOE-encoding |
| `debt_amount` unificado (DebtRatio si falta ingreso, DebtRatio×ingreso si está presente) | **Propuesta, sin validar** | Verificar que no mezcle incorrectamente las filas con ingreso = 0 antes de usarla |
| Interacción edad × utilización | **Detectada descriptivamente, por formalizar** | Probar como feature de interacción explícita en modelo lineal; en árboles debería capturarse sola |
| Usar variables de señal individual débil (DebtRatio, líneas abiertas, inmuebles, dependientes) en escala continua para un modelo lineal | **Probable candidata a descartar** en esa forma | Preferir binning/WOE o dejarlas para un modelo de árboles que capture la no linealidad sin transformación manual |

### Features propuestas (síntesis de ambos EDA)
1. `late_sentinel` + limpieza de 96/98 → `NaN` en las tres columnas de atraso.
2. Banderas `income_missing`, `income_zero`, `dependents_missing`.
3. `debt_ratio_valid` / `debt_amount` unificado (pendiente de validar, ver tabla de hipótesis).
4. Atrasos agregados: total, bandera "algún atraso", máximo.
5. Utilización: tope en 2, logaritmo, bandera utilización > 1.
6. Edad: `NaN` si < 18; tramos o interacción edad × utilización para el baseline lineal.
7. `sin_lineas_credito = (NumberOfOpenCreditLinesAndLoans == 0)` como bandera de riesgo (nuevo, de `01_eda_karla.ipynb`).
8. Discretización/bins de `NumberOfOpenCreditLinesAndLoans` y `NumberRealEstateLoansOrLines` si se usa un modelo lineal, dado su comportamiento no monótono.

### Modelos sugeridos
- **Baseline lineal** (regresión logística) sobre las features transformadas/discretizadas arriba — necesario precisamente porque varias variables no son monótonas.
- **Modelo principal: LightGBM** — captura de forma nativa las relaciones en forma de "U" y la interacción edad × utilización sin ingeniería manual extensa.
- **Validación:** `StratifiedKFold` (repetida) por el desbalance moderado; métrica AUC-ROC. Sin evidencia de drift train/test, por lo que el CV local es representativo del leaderboard.

## Ingeniería de variables y modelos (línea JAE)

El detalle está en la carpeta [`docs/`](docs/):

| Documento | Contenido |
| --- | --- |
| [`docs/plan_ingenieria_variables.md`](docs/plan_ingenieria_variables.md) | Plan de ingeniería de variables derivado de `notebooks/02_eda.ipynb`: principios, imputación del ingreso (⭐ B: edad + decil del monto de deuda; 🔷 C como alternativa), escalado y propuesta por variable. Implementado en `src/features/engineering_jae.py` y `notebooks/03_ingenieria_variables_jae.ipynb`. |
| [`docs/aprendizajes_iteracion_05.md`](docs/aprendizajes_iteracion_05.md) | Aprendizajes de la iteración 05 (`notebooks/05_mejora_modelos_jae.ipynb`): traza de iteraciones, comparación de variables JAE contra FE (`features.prepare()`) y siguientes pasos. |
| [`docs/modelo_ganador.md`](docs/modelo_ganador.md) | 🥈 Explicación sencilla del modelo ganador (2.º lugar): ingeniería de variables, imputación del ingreso, flujo y características del ensamble. |

### Mejor modelo hasta el momento

| Modelo | AUC validación cruzada | Kaggle |
| --- | --- | --- |
| XGBoost 2b (`notebooks/04_modelos_base_jae.ipynb`) | 0.86586 ± 0.00322 | 0.87992 |
| **Entrega B**: mezcla 50/50 de XGBoost JAE y XGBoost FE, 5 semillas cada uno (`notebooks/05_mejora_modelos_jae.ipynb`) | **0.86663** (gana a 2b en los 5 folds) | **0.88012** |
| Entrega C: como la B, con JAE + interacciones (`notebooks/06_interacciones_jae.ipynb`) | 0.86666 (empata con B: +0.00003, 3 de 5 folds) | — |
| Entrega D: como la C, sin `disposable_income_clip` ni `total_late` (25 variables JAE; `notebooks/07_seleccion_catboost_jae.ipynb`) | 0.86676 (+0.00013 sobre B, 4 de 5 folds) | 0.87996 |
| **Entrega F**: 70 % entrega D + 30 % CatBoost, mezcla por rangos (`notebooks/07_seleccion_catboost_jae.ipynb`) | **0.86698** (+0.00035 sobre B, 5 de 5 folds) | **0.88016** (privado 0.86509: 2.º lugar final) |

### Variables de FE incorporadas al conjunto JAE

Según la comparación de la sección 4 de `docs/aprendizajes_iteracion_05.md`,
las variables de FE con información nueva para los árboles eran
`age_x_util`, `late_x_util`, `non_re_lines` y `max_late`. Desde
`notebooks/06_interacciones_jae.ipynb` forman parte del conjunto JAE
(`FeatureConfig(interactions=True)`) y del dataset de entrenamiento
`data/processed/jae/train_jae.parquet` (30 variables):

- Mejoran al XGBoost JAE en +0.00015 (dentro del ruido) y cierran cerca del
  60 % de la distancia con el conjunto FE.
- El modelo las usa mucho (`late_x_util` es la variable de mayor ganancia),
  pero sustituyen a los atrasos y la utilización que ya existían, en lugar de
  aportar información nueva.
- `disposable_income_clip` (ingreso − monto de deuda), la variable nueva con
  más señal del EDA (IV 0.161), se conserva.
