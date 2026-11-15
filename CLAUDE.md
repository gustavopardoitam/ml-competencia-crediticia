# CLAUDE.md

Guía para Claude Code en este repositorio. Mantiene las equivalencias
operativas de `.github/copilot-instructions.md` para que el trabajo con Claude
y Copilot siga las mismas reglas. **La fuente de verdad de las políticas es
`.github/copilot-instructions.md`**; léelo y respétalo íntegramente. Si una
política cambia, actualiza ambos archivos.

## Proyecto

Predicción de `P(SeriousDlqin2yrs = 1)` en competencia crediticia. Métrica:
AUC-ROC. Metodología: CRISP-DM.

## Comandos

- Instalar: `uv sync --frozen`
- Descargar datos: `PYTHONPATH=src uv run python -m etl.download`
- Sincronizar notebook: `uv run jupytext --sync notebooks/01_eda.py`
- Pruebas: `uv run pytest`

## Estructura

- `src/config`, `src/etl`, `src/features`, `src/training`, `src/evaluation`,
  `src/inference`, `src/utils`: código por etapa del pipeline.
- `notebooks/`: EDA (`.ipynb` emparejado con `.py` para revisar diffs).
- `data/raw/`: datos crudos, no se versionan.

## Políticas resumidas

- **Idioma:** todo en español de México (docs, comentarios, docstrings,
  `logging`, excepciones, mensajes de commit). Se conservan en su idioma
  original: identificadores de código, claves JSON/YAML, nombres de archivo,
  comandos y nombres propios.
- **Evidencia:** todo entregable debe tener evidencia verificable (README, PRs,
  commits, logs o pruebas).
- **Seguridad:** nunca exponer secretos ni datos sensibles; el token de Kaggle
  vive en `~/.kaggle/access_token`.
- **Datos:** toda transformación define reglas DQ y cifras de control; evitar
  *data leakage*; experimentos reproducibles (semillas, `uv.lock`).
- **Arquitectura:** registrar decisiones relevantes (ADR) antes de introducir
  patrones nuevos.

### Estándares de Código

- Python 3.11+ compatible.
- PEP 8, Black-compatible y líneas de máximo 88 caracteres.
- Módulos y funciones en `snake_case`, clases en `PascalCase`, constantes en
  `UPPER_SNAKE_CASE`.
- Todos los parámetros y retornos deben tener anotaciones de tipo.
- Usar docstrings estilo Google en funciones y clases públicas.
- Usar `logging`, nunca `print()` para observabilidad de aplicación.
- Capturar excepciones específicas, registrar el error y relanzarlo cuando
  corresponda; nunca ocultar fallos silenciosamente.
- No hardcodear credenciales, secretos ni rutas dependientes del entorno.
- Usar `pytest`, fixtures y mocks para dependencias externas.

#### Convenciones de nombres (PEP 8)

| Elemento | Convención | Ejemplo |
| --- | --- | --- |
| Módulos | `snake_case` | `data_loader.py` |
| Paquetes | `snake_case` | `etl/` |
| Clases | `PascalCase` | `DataPipeline` |
| Funciones | `snake_case` | `load_dataframe()` |
| Variables | `snake_case` | `train_df` |
| Constantes | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| Miembros privados | `_guion_bajo` | `_internal_cache` |

#### Formato de código

- Línea máxima de 88 caracteres (compatible con Black).
- Indentación de 4 espacios, nunca tabs.
- Líneas en blanco: 2 entre definiciones de nivel superior y 1 entre métodos
  de una clase.
- Imports ordenados en tres grupos: biblioteca estándar, terceros y proyecto
  local. Un import por línea.
- Evitar imports con comodín (`from x import *`).

#### Anotaciones de tipo (obligatorias)

```python
# Correcto
def compute_auc(y_true: list[int], y_score: list[float]) -> float:
    """Calcula el AUC-ROC entre etiquetas reales y probabilidades."""

# Incorrecto
def compute_auc(y_true, y_score):
    ...
```

- Todos los parámetros y retornos llevan anotaciones.
- Usar `T | None` u `Optional[T]` para valores nulables.
- Usar `list[T]`, `dict[K, V]` y `tuple[T, ...]` en minúsculas.
- Evitar `Any` sin justificación explícita.

#### Docstrings (estilo Google)

```python
def load_predictions(client_id: int, date: str) -> pd.DataFrame:
    """Carga las predicciones de un cliente para una fecha.

    Args:
        client_id: Identificador único del cliente.
        date: Fecha objetivo en formato ISO (AAAA-MM-DD).

    Returns:
        DataFrame con los registros de predicción.

    Raises:
        ValueError: Si `client_id` es negativo o `date` está malformada.
    """
```

#### Manejo de errores

```python
# Correcto
try:
    result = load_data(path)
except FileNotFoundError:
    logger.error("No se encontró el archivo de datos: %s", path)
    raise

# Incorrecto
try:
    result = load_data(path)
except Exception:
    pass  # Falla silenciosa
```

#### Logging (nunca `print()`)

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Cargando datos desde %s", file_path)
logger.warning("Valores faltantes detectados: %d filas", missing_count)
logger.error("Falló la conexión a la base de datos: %s", str(e))
```

#### Configuración (sin hardcodeo)

```python
# Correcto
DB_HOST = os.environ["DB_HOST"]
DB_PASSWORD = os.environ["DB_PASSWORD"]  # falla si no está definida

# Incorrecto
DB_HOST = "localhost"
DB_PASSWORD = "admin123"
```

- No usar valores por defecto para secretos.

#### Pruebas con pytest

- Usar `pytest` para todas las pruebas.
- Archivos `test_<modulo>.py` y funciones `test_<comportamiento>()`.
- Preferir fixtures sobre `setup`/`teardown`.
- Simular (mock) dependencias externas: bases de datos, S3, APIs, Kaggle.

### Criterios de Calidad

Todo código generado o revisado debe cubrir:

1. PEP 8 y legibilidad.
2. Anotaciones de tipo.
3. Docstrings.
4. Manejo específico de errores.
5. Logging adecuado.
6. Configuración externa.
7. Seguridad y OWASP Top 10.
8. Modularidad.
9. Testabilidad.
10. Reproducibilidad.

## Quality Gates

- [ ] PEP 8 compliance
- [ ] Type annotations
- [ ] Docstrings (Google style)
- [ ] Error handling y logging
- [ ] Sin hardcoding de credenciales
