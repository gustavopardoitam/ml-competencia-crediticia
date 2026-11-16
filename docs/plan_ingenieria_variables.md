# Plan de ingeniería de variables

Insumo para `notebooks/03_ingenieria_variables`. Las cifras provienen de la
sección 10 de `notebooks/02_eda.ipynb` (train: 105,000 filas, sin `Id`
repetidos).

## Estado

Implementado en `notebooks/03_ingenieria_variables_jae.ipynb` con el módulo
`src/features/engineering_jae.py` (pruebas en `tests/test_engineering_jae.py`).
El EDA, el módulo y el notebook 03 están alineados: la tabla de la sección
10.9 del EDA indica dónde vive cada decisión en el módulo.

Diferencias contra el plan original:

- El WoE se implementó con un codificador propio (`WoEEncoder`) en lugar de
  `optbinning`, para no agregar dependencias.
- Los atrasos se recortan a 6 (≈ p99.9).
- La imputación del ingreso cambió: ⭐ B es la recomendada, 🔷 C la alternativa
  y A quedó solo como experimento de referencia.
- El ingreso = 1 se trata como ingreso 0: indicador `income_placeholder`,
  fuera del ajuste de las medianas y con `DebtRatio` leído como monto.
- Se agregó el ingreso por dependiente (`per_dependent_<ingreso>`).

Resultados principales (AUC, validación cruzada de 5 pliegues):
regresión logística 0.700 con variables crudas → **0.8568** con la propuesta;
árboles 0.8622 → **0.8626**.

- Las estrategias de imputación del ingreso empatan en AUC: en la regresión
  logística (global, A, B, C y B + C) las diferencias son ≤ 0.0002; en
  árboles, ≤ 0.0004 (C queda 0.0004 arriba de B, dentro del ruido). B se
  elige por ser la que imputa con menor error.
- El ingreso por dependiente no mejora la regresión logística (0.0000 al
  quitarlo) ni tiene efecto consistente en árboles. Se conserva en el
  dataset.

El detalle y los hallazgos están en la sección 9 del notebook.

## Principios

1. **Orden de aplicación:** limpieza fija → imputación → transformación →
   escalado.
2. **Transformaciones fijas** (recortes con umbral fijo, códigos 96/98 a
   `NaN`, indicadores): se aplican fuera de la validación cruzada. Viven en
   `clean_raw()` y `add_fixed_features()` de
   `src/features/engineering_jae.py`, un módulo independiente del
   `features.py` de la rama `feature/features-separabilidad`.
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

## Imputación del ingreso: ⭐ B (recomendada) y 🔷 C (alternativa)

Comparación de estrategias (sección 10.3 d del EDA): se oculta el ingreso del
30 % de los clientes que lo tienen y se mide el error de cada estrategia.

| Estrategia | Grupos | Error (MAE en log) | Mejora vs global |
| --- | --- | --- | --- |
| Mediana global (línea base) | 1 | 0.577 | — |
| Mediana por grupo de edad | 11 | 0.543 | 5.9 % |
| Edad + utilización de la línea de crédito (quintil) | 55 | 0.538 | 6.7 % |
| Edad + dependientes (0, 1, 2+) | 33 | 0.535 | 7.3 % |
| Edad + n.º de líneas de crédito abiertas (quintil) | 55 | 0.527 | 8.6 % |
| 🔷 C (alternativa): edad + líneas de crédito + dependientes | 162 | 0.521 | 9.7 % |
| A (descartada): edad + n.º de créditos hipotecarios | 33 | 0.515 | 10.6 % |
| Mediana por decil del monto de deuda | 10 | 0.498 | 13.7 % |
| **⭐ B (recomendada): edad + decil del monto mensual de deuda** | 109 | **0.484** | **16.1 %** |

**⭐ Estrategia B → `income_imp_age_debt`.** Es la recomendada y la que el
módulo usa por omisión:

1. Se calcula el monto mensual de deuda (`debt_amount`, transformación fija):
   `DebtRatio × MonthlyIncome` si hay ingreso; `DebtRatio` tal cual si falta
   el ingreso, es 0 o es el valor de relleno 1.
2. El monto se corta en deciles, con cortes aprendidos en el pliegue.
3. Se imputa la mediana del grupo edad × decil de deuda; si el grupo no
   existe, la mediana global.

Por qué B como recomendada:

- Es la que mejor imputa: 16.1 % de mejora, contra 10.6 % de A y 9.7 % de C.
  El monto de deuda es la variable más relacionada con el ingreso
  (Spearman 0.50).
- La hipótesis en la que se apoya está bien respaldada (sección 10.3 b): sin
  ingreso, el 100 % de los `DebtRatio` son enteros y su distribución coincide
  con la del monto de deuda de los clientes con ingreso.
- El monto de deuda siempre está informado.

**🔷 Estrategia C → `income_imp_age_lines_dep`.** Se conserva como segunda
variable de ingreso: no depende de la hipótesis sobre `DebtRatio` ni del
monto de deuda, y se parece poco a B (Spearman 0.65), así que puede aportar
información distinta. Imputa con respaldo:

1. Mediana del grupo edad × quintil de líneas abiertas × dependientes
   (0, 1, 2+), si el grupo tiene al menos 30 clientes con ingreso.
2. Si no (o si faltan los dependientes), mediana de edad × líneas.
3. Si tampoco, mediana del grupo de edad y, al final, mediana global.

Aplicadas a los 20,828 clientes sin ingreso: la mediana imputada es 4,502 con
B, 5,000 con C y 5,416 con A. B es la más repartida (desviación 1,991 contra
1,297 de C). En C, el 86.6 % se imputa con el grupo completo y el 13.4 % (sin
dependientes informados) con edad × líneas.

**Experimento:** comparar por AUC usar solo B, solo C o ambas (con la mediana
global y A como referencia).

## Escalado propuesto

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
| `MonthlyIncome` | ⭐ `income_imp_age_debt` (estrategia B) + `income_missing` | Recortar a p99 (25,000) + `income_zero` + `income_placeholder` (ingreso = 1) | `log1p`; derivadas: ingreso disponible e ingreso por dependiente | `StandardScaler` | 🔷 `income_imp_age_debt` (B) como segunda variable; solo C, solo B o ambas |
| `NumberOfOpenCreditLinesAndLoans` | — | Recortar a p99 (24) | — | `StandardScaler` | — |
| `NumberRealEstateLoansOrLines` | — | Recortar a 5 | — | `StandardScaler` | — |
| `NumberOfDependents` | Mediana por grupo de edad + `dependents_missing` | Recortar a 5 | — | `StandardScaler` | Imputar 0 |

Grupos de edad: cortes `[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120]`.
Grupos de la estrategia C: quintiles de líneas abiertas (cortes aprendidos
en el pliegue) y dependientes en 0, 1 o 2+; mínimo de 30 clientes por grupo.

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

1. `src/features/engineering_jae.py`:
   - Limpieza y recortes fijos: utilización 2, atrasos 6, ingreso p99
     (25,000), líneas abiertas p99 (24), hipotecas 5, dependientes 5.
   - Indicadores: `late_sentinel`, `income_missing`, `income_zero`,
     `income_placeholder` (ingreso = 1), `dependents_missing`, `util_gt1`.
   - `GroupMedianImputer`: transformador compatible con `scikit-learn`
     (`fit`/`transform`) que imputa la mediana por grupo, con respaldo a la
     mediana global para grupos sin datos. Se usa para ⭐
     `income_imp_age_debt` (B), `income_imp_age_mortgage` (A, solo de
     referencia) y dependientes (edad); con niveles de respaldo y tamaño
     mínimo de grupo para 🔷 `income_imp_age_lines_dep` (C).
   - Variables fijas `debt_amount` y `debt_ratio_valid`.
   - Función que construye el `ColumnTransformer` según el tipo de modelo
     (lineal o de árboles).
   - Variables nuevas: `disposable_<ingreso>` (ingreso disponible),
     `per_dependent_<ingreso>` (ingreso por dependiente), `util_zero` y
     `thin_file`.
   - Representación WoE con el codificador propio `WoEEncoder`.
2. `tests/test_engineering_jae.py`: pruebas del imputador (no usa datos del
   pliegue de validación, no deja `NaN`, respeta el respaldo por niveles y el
   global), del ingreso de relleno, de las variables derivadas y de los
   recortes.
3. Notebook con los experimentos:
   - Ingreso: solo ⭐ B, solo 🔷 C o ambas (mediana global y A como
     referencia).
   - Escalado: `StandardScaler` contra `RobustScaler` contra
     `QuantileTransformer`.
   - Atrasos: conteo transformado contra binarizado.
   - Representación para la regresión logística: escaladas contra WoE
     contra *splines*.
   - Con y sin ingreso disponible, ingreso por dependiente e indicadores de
     perfil.
   - Modelos: regresión logística y un modelo de árboles.
4. Cifras de control en cada paso: filas de entrada y salida, y cero `NaN`
   después de imputar (modelos lineales).
