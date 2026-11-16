# Plan de ingeniería de variables

Insumo para `notebooks/03_ingenieria_variables`. Las cifras provienen de la
sección 10 de `notebooks/02_eda.ipynb` (train: 105,000 filas, sin `Id`
repetidos).

## Estado

Implementado en `notebooks/03_ingenieria_variables_jae.ipynb` con el módulo
`src/features/engineering_jae.py` (pruebas en `tests/test_engineering_jae.py`).
Diferencias contra este plan:

- El WoE se implementó con un codificador propio (`WoEEncoder`) en lugar de
  `optbinning`, para no agregar dependencias.
- Los atrasos se recortan a 6 (≈ p99.9).

Resultados principales (AUC, validación cruzada de 5 pliegues):
regresión logística 0.700 con variables crudas → **0.8565** con la propuesta;
árboles 0.8622 → **0.8629**. El detalle y los hallazgos están en la sección 9
del notebook.

**Pendientes (hallazgos posteriores al notebook 03):**

- Tratar el ingreso = 1 como el ingreso = 0 (indicador propio y fuera del
  ajuste de las medianas).
- Agregar `income_per_dependent` = ingreso imputado / (1 + dependientes).

## Principios

1. **Orden de aplicación:** limpieza fija → imputación → transformación →
   escalado.
2. **Transformaciones fijas** (recortes con umbral fijo, códigos 96/98 a
   `NaN`, indicadores): se aplican fuera de la validación cruzada. Base:
   `clean()` y `add_features()` de `src/features/features.py` (rama
   `feature/features-separabilidad`).
3. **Transformaciones aprendidas** (medianas, escaladores, percentiles): se
   ajustan **solo con el pliegue de entrenamiento** dentro de un `Pipeline`.
   Nunca se ajustan sobre `train` completo ni sobre `test`.
4. Validación cruzada estratificada de 5 pliegues, semilla 42, métrica AUC-ROC.
5. En `test` no se elimina ninguna fila.

## Hallazgos que motivan el plan

| Hallazgo | Evidencia | Sección |
| --- | --- | --- |
| Ingreso muy disperso | CV 2.3; media 6,646 contra mediana 5,400; máximo 120 veces el p99 | 10.2 |
| `NumberOfDependents` solo falta cuando también falta `MonthlyIncome` | 2,771 de 2,771 filas | 10.3 a |
| Sin ingreso, `DebtRatio` siempre está informado y es un monto | 20,828 de 20,828 informados; 100 % enteros; mediana 1,158 contra 0.30 | 10.3 b |
| El monto de deuda ayuda a inferir el ingreso | Spearman 0.50 con el ingreso | 10.3 b |
| El ingreso cambia con la edad y con las hipotecas | 45–50 años: 4,789 sin hipoteca contra 7,916 con 2 o más | 10.3 c |
| Utilización > 10 son errores de captura | Morosidad 40 % en (1, 2] y 6 % en > 10 | 10.4 |
| Atrasos 96/98 son códigos, no conteos | 197 filas en train y 72 en test; atraso real máximo 15 | 10.4 |
| Atrasos casi binarios | 84–95 % ceros (IQR = 0, `RobustScaler` no sirve) | 10.1 |
| Monto de deuda muy disperso y con relación en U con la morosidad | CV 2.1; morosidad 8.4 % (46–1,129), 5.2 % (centro), 8.9 % (> 4,277); deuda 0: 8.6 % | 10.3 e |
| Más dependientes, más riesgo, sobre todo con pocos recursos | Morosidad 5.9 % (0) → 10.1 % (5+); en el cuartil de ingreso más bajo 7.6 % → 19.2 %; ingreso por dependiente IV 0.125 | 10.3 f |
| Ingreso = 1 es un valor de relleno | 423 clientes; `DebtRatio` mediano 625, 94 % > 1 | 10.3 e |
| El jugo en los datos está repartido | 7 de 10 componentes para el 80 %; PC1 (riesgo) explica 19.8 % y logra AUC 0.82 solo | 10.7 |

## Imputación del ingreso: dos estrategias → dos variables de ingreso

| Estrategia | Grupos | Error (MAE en log) | Mejora vs global |
| --- | --- | --- | --- |
| Mediana global (línea base) | 1 | 0.577 | — |
| Mediana por grupo de edad | 11 | 0.543 | 5.9 % |
| **⭐ A: edad + n.º de créditos hipotecarios** | 33 | **0.515** | **10.6 %** |
| Mediana por decil del monto de deuda | 10 | 0.498 | 13.7 % |
| **🔷 B: edad + decil del monto mensual de deuda** | 109 | **0.484** | **16.1 %** |

Se construyen **las dos** como variables del dataset:

| Variable | Estrategia | Ventaja | Riesgo |
| --- | --- | --- | --- |
| ⭐ `income_imp_age_mortgage` | A (principal) | Fácil de explicar; variables siempre informadas | Imputa con menos precisión |
| 🔷 `income_imp_age_debt` | B (alternativa) | Imputa mejor; el monto de deuda es el mejor predictor del ingreso (Spearman 0.50) | Depende de que, sin ingreso, `DebtRatio` sea el monto de deuda |

Monto mensual de deuda (`debt_amount`, transformación fija):
`DebtRatio × MonthlyIncome` si el ingreso es > 0; `DebtRatio` tal cual si
falta el ingreso o es 0.

Aplicadas a los 20,828 clientes sin ingreso, A y B tienen Spearman 0.84 y
difieren más de 30 % en el 11.9 % de los casos: B aporta información que A
no tiene. B está más repartida (1,200 a 11,359) y se parece más a la forma
del ingreso real. Ambas llevan el indicador `income_missing`.

**Experimento:** comparar por AUC usar solo A, solo B o ambas (la
regularización L1/L2 maneja la correlación entre ellas).

## Escalado que vamos a usar

Solo para la regresión logística con regularización; los modelos de árboles no
se escalan.

| Técnica | Papel |
| --- | --- |
| `log1p` | Paso previo para ingreso, `DebtRatio`, monto de deuda y atrasos |
| `StandardScaler` | **Opción base** para todas las variables continuas |
| `RobustScaler` | Comparación (no en atrasos) |
| `QuantileTransformer` | Comparación |

## Propuesta por variable

| Variable | Faltantes | Limpieza fija | Transformación | Escalado | Alternativa a probar |
| --- | --- | --- | --- | --- | --- |
| `RevolvingUtilizationOfUnsecuredLines` | — | Recortar a [0, 2] + `util_gt1` | — | `StandardScaler` | `QuantileTransformer` |
| `age` | Mediana (1 fila) | Edad < 18 → `NaN` | — | `StandardScaler` | — |
| `NumberOfTime30-59DaysPastDueNotWorse` | 96/98 → 0 + `late_sentinel` | 96/98 → `NaN` | Recortar a p99.9 + `log1p` | `StandardScaler` | Binarizar (0 / ≥ 1) |
| `NumberOfTime60-89DaysPastDueNotWorse` | Igual | Igual | Igual | `StandardScaler` | Igual |
| `NumberOfTimes90DaysLate` | Igual | Igual | Igual | `StandardScaler` | Igual |
| `DebtRatio` | — | Separar en `debt_ratio_valid` (≤ 10) y `debt_amount` | `log1p` | `StandardScaler` | `RobustScaler` |
| `MonthlyIncome` | **Dos variables:** ⭐ `income_imp_age_mortgage` y 🔷 `income_imp_age_debt` + `income_missing` | Recortar a p99 (25,000) + `income_zero` | `log1p` a ambas | `StandardScaler` | Solo A, solo B o ambas |
| `NumberOfOpenCreditLinesAndLoans` | — | Recortar a p99 (24) | — | `StandardScaler` | — |
| `NumberRealEstateLoansOrLines` | — | Recortar a 5 | — | `StandardScaler` | — |
| `NumberOfDependents` | Mediana por grupo de edad + `dependents_missing` | Recortar a 5 | — | `StandardScaler` | Imputar 0 |

Grupos de edad: cortes `[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120]`.
Hipotecas para agrupar: `NumberRealEstateLoansOrLines` recortado a 2.

**Modelos de árboles:** misma limpieza fija e indicadores, sin escalado.
Comparar imputación contra dejar `NaN` (XGBoost y
`HistGradientBoostingClassifier` lo manejan directamente).

**PCA:** no se usa para reducir dimensiones (la información está repartida).
Sí confirma que el eje de riesgo (utilización + atrasos + edad) concentra la
señal, así que esas variables son la prioridad de la ingeniería de variables.

## Alternativas adicionales (sección 10.8 del EDA)

| Alternativa | Evidencia | Recomendación |
| --- | --- | --- |
| **WoE / IV** | IV: utilización 1.12, atrasos 30–59 0.64, 90+ 0.59, 60–89 0.38, edad 0.27; el resto < 0.1. Captura la curva de la utilización (sube hasta 1–2, baja después; utilización 0 es menos segura que 0–0.1) | ✅ **Prioridad alta:** enfoque paralelo para la regresión logística, con `optbinning` (cortes y WoE aprendidos en cada pliegue) |
| **Ingreso disponible** (`ingreso − debt_amount`) | IV 0.161 contra 0.088 del ingreso; morosidad monótona de 11.1 % a 3.3 % por decil | ✅ **Prioridad alta:** variable nueva; con ingreso imputado cuando falta |
| Imputación del ingreso con modelo | Con `DebtRatio` mejora 37.1 %, pero es engañoso (la razón solo existe con ingreso); con monto de deuda, 17.9 % (apenas mejor que B) | ❌ No prioritaria |
| *Splines* (`SplineTransformer`) | Relaciones no lineales en utilización y edad | Experimento secundario |
| Indicadores de perfil | `util_zero`: 7,652 clientes con 3.0 % de morosidad; `thin_file`: 316 con 1.6 % (resto: 6.7 %) | Experimento secundario (muy barato) |

## Entregables del notebook 03

1. `src/features/`:
   - Recortes faltantes en `features.py`: ingreso p99, líneas abiertas p99,
     hipotecas 5, dependientes 5.
   - `GroupMedianImputer`: transformador compatible con `scikit-learn`
     (`fit`/`transform`) que imputa la mediana por grupo, con respaldo a la
     mediana global para grupos sin datos. Se usa para ⭐
     `income_imp_age_mortgage` (edad + hipotecas), 🔷 `income_imp_age_debt`
     (edad + decil de deuda, con cortes aprendidos en el pliegue) y
     dependientes (edad).
   - Variable fija `debt_amount` (ya existe en `features.py`).
   - Función que construye el `ColumnTransformer` según el tipo de modelo
     (lineal o de árboles).
   - Variables nuevas: `disposable_income` (ingreso disponible),
     `util_zero` y `thin_file`.
   - Representación WoE con `optbinning` (agregar la dependencia).
2. `tests/`: pruebas del imputador (no usa datos del pliegue de validación,
   no deja `NaN`, respeta el respaldo global) y de los recortes.
3. Notebook con los experimentos:
   - Ingreso: solo ⭐ A, solo 🔷 B o ambas (mediana global como línea
     base).
   - Escalado: `StandardScaler` contra `RobustScaler` contra
     `QuantileTransformer`.
   - Atrasos: conteo transformado contra binarizado.
   - Representación para la regresión logística: escaladas contra WoE
     contra *splines*.
   - Con y sin ingreso disponible e indicadores de perfil.
   - Modelos: regresión logística L1/L2 y un modelo de árboles.
4. Cifras de control en cada paso: filas de entrada y salida, y cero `NaN`
   después de imputar (modelos lineales).
