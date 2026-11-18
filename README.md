# ML Championship: competencia crediticia

Predicción de P(SeriousDlqin2yrs = 1). Métrica: AUC-ROC.

## Setup
`uv sync --frozen`. Los datos van en `data/raw/` y no se versionan.

## Reproducir el EDA
1. `git clone` y `uv sync`.
2. Acepta las reglas de la competencia en Kaggle y guarda tu token en `~/.kaggle/access_token` (nunca en el repo).
3. `PYTHONPATH=src uv run python -m etl.download`
4. Abre `notebooks/01_eda.ipynb` con el kernel `.venv`. El `.py` emparejado sirve para revisar diffs; sincroniza con `uv run jupytext --sync notebooks/01_eda.py`.

## Validar un modelo

Todo experimento (combinación de features + modelo) se mide con la misma partición de cross-validation y se registra en una bitácora compartida, para poder comparar resultados de forma justa entre todo el equipo.

### 1. Evaluar y registrar

```python
import pandas as pd
from lightgbm import LGBMClassifier
from features.features import prepare  # o el conjunto de variables que usaron
from evaluation import run_experiment

train = pd.read_csv("data/raw/train.csv")
y = train["SeriousDlqin2yrs"]
X = prepare(train)  # DataFrame numérico, sin Id ni target; los NaN se dejan tal cual

folds = run_experiment(
    "baseline_lightgbm", LGBMClassifier(random_state=985), X, y,
    features_desc="prepare() - 28 columnas",
    notas="piso de comparación",
)
```

`run_experiment` entrena el modelo en 5 folds estratificados (siempre los mismos, ver `RANDOM_STATE` en `src/evaluation/cv.py`), imprime el AUC, agrega una fila a `reports/experiments.csv` y devuelve el AUC de cada fold. Acepta cualquier modelo con `.fit`/`.predict_proba`: `LGBMClassifier`, `LogisticRegression`, un `Pipeline`, un `VotingClassifier`, etc.

Si un paso usa el target (WOE, target encoding, selección de variables), debe ir **dentro** de un `Pipeline`: así se ajusta solo con el train de cada fold y no hay leakage.

Piezas sueltas, por si se necesitan por separado:
- `evaluate(model, X, y)` → `(auc_media, auc_std)`, sin registrar.
- `evaluate_folds(model, X, y)` → AUC de cada fold, sin registrar.
- `log_experiment(nombre, features, modelo, cv_auc_mean, cv_auc_std, kaggle_score=None, notas="")` → registra a mano.

Para revisar la bitácora:

```python
from evaluation import read_log
read_log().sort_values("cv_auc_mean", ascending=False)
```

### 2. Comparar experimentos

Cambia **una sola cosa** entre un experimento y el siguiente (o las features, o el modelo, no ambos), para saber qué causó la diferencia.

Como todos los experimentos usan la misma partición, conviene compararlos **fold por fold**. Es más sensible que comparar la mejora contra `cv_auc_std`, que incluye la variación de qué tan difícil es cada fold:

```python
from evaluation import paired_comparison

paired_comparison({"lightgbm_nuevo": folds_nuevo}, reference=folds_baseline)
#                 mejora_media  std_pareada  folds_que_mejoran
# lightgbm_nuevo       0.00299      0.00070                  5
```

Una mejora es confiable si `mejora_media` es varias veces `std_pareada` y mejora en (casi) todos los folds. Si se eligen hiperparámetros o pesos con esta misma partición, la mejora sale algo optimista: para búsquedas, usa otra partición (`get_cv(random_state=42)`) y mide al final con la estándar.

### 3. Generar el submission y anotar el score de Kaggle

```python
from evaluation import set_kaggle_score
from inference import make_submission

test = pd.read_csv("data/raw/test.csv")
sample = pd.read_csv("data/raw/sample_submission.csv")

make_submission(
    LGBMClassifier(random_state=985), X, y, prepare(test), test["Id"], sample,
    "submission_baseline_lightgbm.csv",
)
# ... subir data/processed/submission_baseline_lightgbm.csv a Kaggle, y luego:
set_kaggle_score("baseline_lightgbm", 0.87884)
```

`make_submission` entrena con el 100% de `train`, predice `test` (que solo se transforma, sin volver a ajustar nada), valida que el formato y los `Id` coincidan con `sample_submission.csv` y guarda el CSV en `data/processed/` (fuera de git). `set_kaggle_score` anota el score en el registro más reciente con ese nombre.

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
