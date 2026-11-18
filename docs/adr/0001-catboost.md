# ADR 0001: agregar CatBoost como dependencia

- **Estado:** aceptada
- **Fecha:** 2026-11-17
- **Ámbito:** modelado (línea JAE)

## Contexto

Al cierre de la competencia, el modelo había llegado a una meseta de AUC
≈ 0.8668 en validación cruzada:

- La búsqueda de hiperparámetros de XGBoost (30 configuraciones) no superó a
  los parámetros 2b.
- Los conjuntos de variables JAE y FE producen modelos casi idénticos
  (Spearman 0.998 entre sus predicciones OOF).
- Las mezclas con LightGBM, XGBoost poco profundo, bosque aleatorio,
  ExtraTrees y la regresión logística no mejoraron la entrega D.

Hacía falta un modelo de *boosting* con errores distintos a los de XGBoost.
CatBoost usa árboles simétricos y un esquema de *boosting* ordenado, por lo
que suele diferir de XGBoost y LightGBM.

## Decisión

Se agrega `catboost>=1.2.10` a `pyproject.toml` y `uv.lock` (`uv add
catboost`). Se usa como miembro de mezcla, no como modelo principal.

## Evidencia

| Experimento | AUC validación cruzada | Kaggle (público) |
| --- | --- | --- |
| `jae_9_entrega_d_mezcla` (sin CatBoost) | 0.86676 | 0.87996 |
| `jae_10_catboost_seleccion_25` (CatBoost solo) | 0.86618 | — |
| `jae_11_entrega_f_catboost_w30` (70 % D + 30 % CatBoost) | 0.86698 | 0.88016 |

- La entrega F gana a la D en los 5 folds de la partición del equipo
  (+0.00022) y en 4 de 5 de una partición independiente (semilla 2024,
  +0.00012).
- Fue la mejor entrega del equipo en el leaderboard público y mantuvo el
  tercer lugar en el privado.

## Consecuencias

- Una dependencia más (incluye `graphviz` y `plotly` como dependencias
  transitivas).
- CatBoost escribe artefactos de entrenamiento en `catboost_info/`; se agregó a
  `.gitignore`.
- Pendiente: pasar el experimento de scripts exploratorios a un notebook con
  trazabilidad (notebook 07).
