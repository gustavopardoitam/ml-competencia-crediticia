# Aprendizajes de la iteración 05: entregas A y B

Resumen de lo aprendido al mejorar el XGBoost 2b del notebook 04. El detalle,
con la trazabilidad de cada iteración, está en
`notebooks/05_mejora_modelos_jae.ipynb`.

## Contexto

| Referencia | AUC |
| --- | --- |
| XGBoost 2b (notebook 04), validación cruzada | 0.86586 ± 0.00322 |
| XGBoost 2b, Kaggle (puntaje público) | 0.87992 |
| Equipo en primer lugar al iniciar la iteración | 0.88010 |
| Equipo en primer lugar al cerrar la iteración | 0.88035 |

**Esquema de validación:** partición del equipo (`get_cv()`, 5 folds
estratificados, semilla 985) y comparación pareada fold por fold. Todo lo que
se aprende de los datos se ajusta dentro de cada fold, sin data leakage.

## 1. Exploración previa: dónde invertir el esfuerzo

Antes de construir el notebook se midieron, de forma rápida, cuatro líneas de
mejora contra el XGBoost 2b (número de árboles fijo, sin registrar en la
bitácora).

| Línea | Variante | ΔAUC vs 2b | Folds en que gana |
| --- | --- | --- | --- |
| Quitar variables | Sin los 7 indicadores con importancia casi nula | −0.00043 | 1/5 |
| Quitar variables | Solo las 10 variables crudas | −0.00029 | 1/5 |
| Ingeniería de variables | Mismo modelo con las variables de `features.prepare()` | +0.00037 | 4/5 |
| Hiperparámetros | `learning_rate=0.01` con 800 árboles | +0.00036 | 5/5 |
| Hiperparámetros | Promedio de 5 semillas | +0.00026 | 4/5 |
| Otros modelos | LightGBM con las variables JAE | −0.00028 | 1/5 |
| Combinación | XGBoost JAE + XGBoost FE, 5 semillas cada uno | +0.00056 | 5/5 |

**Conclusiones de la exploración:**

- **Quitar variables empeora el AUC.** Los indicadores con importancia casi
  nula aportan un poco en conjunto, y los árboles ya ignoran lo que no sirve.
- **Las familias de variables nuevas ya estaban agotadas.** En
  `03_exploracion_manual` (rama `feature/gustavo-ingenieria-variables`), la
  cohorte por edad, los picos y la redondez, la exposición, el patrón de
  faltantes, la frecuencia de ingreso y el riesgo por segmento quedaron
  dentro del ruido.
- **LightGBM solo no supera a XGBoost**; sirve, en todo caso, para combinar.
- **Las líneas prometedoras** eran hiperparámetros baratos (`learning_rate`
  bajo y semillas) y combinar los dos conjuntos de variables.

## 2. Traza de iteraciones

Cada iteración cambia una sola cosa respecto a la anterior.

| Iteración | Experimento | Cambio respecto a la previa | AUC | Δ vs previa | Folds en que gana | Δ vs 2b |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `jae_2b_xgboost_ajustado` | Reproducción del punto de partida | 0.86586 | — | — | — |
| 1 | `jae_3a_xgboost_lr001` | `learning_rate` 0.05 → 0.01 | 0.86595 | +0.00009 | 3/5 | +0.00009 |
| **2 · Entrega A** | `jae_3b_entrega_a_semillas` | Árboles fijos (764) con todo el fold + 5 semillas | 0.86638 | +0.00043 | 4/5 | +0.00052 |
| 3 | `jae_4_xgboost_fe_semillas` | Variables JAE → FE | 0.86663 | +0.00025 | 4/5 | +0.00077 |
| **4 · Entrega B** | `jae_5_entrega_b_mezcla` | Mezcla 50/50 de la entrega A con la iteración 3 | 0.86663 | +0.00025 (vs A) | 5/5 | +0.00077 |

Hiperparámetros comunes (experimento 2b): `max_depth=6`,
`min_child_weight=10`, `subsample=0.6`, `colsample_bytree=0.7`,
`reg_lambda=10`, `gamma=0.5`.

## 3. Aprendizajes

1. **La entrega B es la mejor:** +0.00077 sobre 2b, gana en los 5 folds y
   tiene el menor error estándar de la diferencia (0.00005).
2. **Bajar `learning_rate` por sí solo no ayudó** (+0.00009, dentro del
   ruido). El *early stopping* pasó de 158 a 764 árboles sin mejora real.
3. **La mejora de la entrega A vino de entrenar con todo el fold.** Con un
   número fijo de árboles, el modelo ya no reserva el 10 % del fold de
   entrenamiento para el *early stopping*. Cada semilla sola logra entre
   0.86626 y 0.86635; el promedio de las 5 apenas sube a 0.86638.
4. **El promedio de semillas aporta poco** con `learning_rate=0.01`: las
   semillas dan resultados casi idénticos porque el modelo ya es estable.
5. **Las variables de `features.prepare()` rinden un poco mejor que las JAE**
   con el mismo modelo (+0.00025, 4 de 5 folds).
6. **La mezcla no supera al modelo FE solo** (0.86663 en ambos). Los dos
   modelos predicen casi lo mismo (correlación de Spearman de 0.998), así que
   hay poca diversidad que aprovechar. Lo que aporta es estabilidad: es la
   única variante que gana en los 5 folds.
7. **Las mejoras son pequeñas frente a la variabilidad entre folds**
   (≈ 0.003), pero se distinguen del ruido gracias a la comparación pareada.
8. **La validación cruzada y Kaggle no están en la misma escala:** el 2b
   obtuvo 0.86586 en validación cruzada y 0.87992 en Kaggle (+0.014), igual
   que el LightGBM del equipo (+0.015). Las decisiones se toman con la
   validación cruzada; Kaggle solo confirma.

## 4. Comparación de variables: JAE contra FE

Comparación fila por fila sobre los 105,000 clientes de `train`, con los dos
conjuntos tal como entran a los modelos del notebook 05:

- **JAE:** 23 variables de `src/features/engineering_jae.py`
  (`FeatureConfig(for_trees=True, impute_for_trees=False)`).
- **FE:** 28 variables de `src/features/features.py` (`prepare()`).

### 4.1 Coincidencias exactas (mismo nombre, mismos valores)

| Variable | Qué es |
| --- | --- |
| `age` | Edad (`NaN` si es menor de 18) |
| `util_clip` | Utilización recortada a 2 |
| `debt_amount` | Monto mensual de deuda |
| `total_late` | Suma de los tres atrasos |
| `late_severity` | 30–59 + 2 × 60–89 + 3 × 90+ |
| `late_sentinel` | 1 si tenía códigos 96/98 |
| `income_missing` | 1 si falta el ingreso |
| `income_zero` | 1 si el ingreso es 0 |
| `dependents_missing` | 1 si faltan los dependientes |
| `util_gt1` | 1 si la utilización supera 1 |

En `total_late` y `late_severity` difiere el 0.1 % de las filas, porque JAE
recorta los atrasos a 6 y FE no.

### 4.2 Equivalentes con otro nombre (misma información, pequeñas diferencias)

| JAE | FE | Diferencia |
| --- | --- | --- |
| `income_clip` | `MonthlyIncome`, `log_income` | JAE recorta a 25,000 (0.9 % de filas); FE usa el valor crudo y su logaritmo |
| `per_dependent_income_clip` | `income_per_dep` | Mismo cálculo; JAE recorta ingreso y dependientes (1.1 % de filas) |
| `debt_ratio_valid` | `debt_ratio_valid` | Mismo nombre, distinto tratamiento: sin ingreso, JAE pone 0 y FE deja `NaN`; además JAE trata el ingreso = 1 como "sin ingreso" |
| `late_30_59`, `late_60_89`, `late_90` | Las tres columnas originales de atraso | JAE pone los códigos 96/98 en 0 y recorta a 6; FE los deja en `NaN` |
| `open_lines`, `real_estate`, `dependents` | `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines`, `NumberOfDependents` | JAE recorta a 24, 5 y 5; FE usa el valor crudo |

Para XGBoost son prácticamente la misma variable: tienen correlación de
Spearman de 1.0, y a un árbol solo le importa el orden de los valores, no la
escala.

### 4.3 Solo en JAE

| Variable | Qué es |
| --- | --- |
| `disposable_income_clip` | Ingreso disponible = ingreso − monto de deuda (IV 0.161 en el EDA) |
| `income_placeholder` | 1 si el ingreso es 1 (valor de relleno) |
| `util_zero` | 1 si la utilización es exactamente 0 |
| `thin_file` | Expediente delgado: sin atrasos, sin utilización y 0 o 1 crédito abierto |

### 4.4 Solo en FE

| Variable | Qué es | ¿Información nueva para los árboles? |
| --- | --- | --- |
| `age_x_util` | Edad × utilización | Sí, es una interacción |
| `late_x_util` | Total de atrasos × utilización | Sí, es una interacción |
| `non_re_lines` | Líneas abiertas − créditos hipotecarios | Sí |
| `max_late` | Máximo de los tres atrasos | Sí |
| `DebtRatio` | Razón de deuda cruda (mezcla razón y monto) | Parcial: JAE la separa en `debt_ratio_valid` y `debt_amount` |
| `RevolvingUtilizationOfUnsecuredLines` | Utilización cruda, sin recortar | Parcial: conserva los valores mayores a 2, que JAE recorta |
| `any_late` | 1 si tuvo algún atraso | No: es `total_late > 0` |
| `log_debt_amount` | Logaritmo de `debt_amount` | No: el árbol ve el mismo orden |

### 4.5 Lectura

- **La ventaja propia de JAE es el ingreso disponible**, la variable nueva con
  más señal del EDA. Los tres indicadores (`income_placeholder`, `util_zero`,
  `thin_file`) aportan poco a XGBoost.
- **La ventaja de FE son las interacciones** (`age_x_util`, `late_x_util`),
  `non_re_lines` y `max_late`. Es la explicación más probable de que FE
  rindiera +0.00025 más que JAE con el mismo modelo.

## 5. Siguientes pasos

1. **Subir la entrega B a Kaggle** y registrar su puntaje en la columna
   `kaggle_score` de `reports/experiments.csv`; después, la entrega A si hay
   envíos disponibles.
2. **Unir lo mejor de ambos conjuntos:** agregar a JAE `age_x_util`,
   `late_x_util`, `non_re_lines` y `max_late`, conservar el ingreso
   disponible y medir si un solo conjunto iguala o supera a la mezcla.
3. **Buscar diversidad real para la mezcla:** CatBoost, o XGBoost con árboles
   poco profundos (profundidad 2–3) y muchas iteraciones.
4. **Validar con más folds** (10 folds o validación repetida) antes de elegir
   entre variantes que difieren en 0.0002 a 0.0005.
5. **No perseguir el puntaje público envío por envío:** la distancia con el
   primer lugar (0.00043) es menor que el ruido del puntaje público.

## Reproducibilidad

| Elemento | Ubicación |
| --- | --- |
| Notebook | `notebooks/05_mejora_modelos_jae.ipynb` (emparejado con `.py`) |
| Combinación de modelos | `combinar_resultados()` en `src/evaluation/oof_jae.py` |
| Bitácora | `reports/experiments.csv` (experimentos `jae_3a` a `jae_5`) |
| Traza de iteraciones | `data/processed/jae/traza_05.json` (no se versiona) |
| Entregas | `data/processed/jae/submission_a_semillas.csv` y `submission_b_mezcla.csv` (no se versionan) |
| Predicciones OOF | `data/processed/jae/oof_mejora_05.parquet` (no se versiona) |

Los archivos de `data/` se regeneran ejecutando el notebook 05.
