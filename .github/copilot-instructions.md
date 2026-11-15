# Instrucciones para GitHub Copilot

Proyecto de ML: predicción de `P(SeriousDlqin2yrs = 1)` en competencia
crediticia. Métrica: AUC-ROC. Estas reglas son las mismas que se aplican en
`CLAUDE.md`; si cambia una, se actualiza la otra.

## Contexto del proyecto

- Python 3.11+ gestionado con `uv` (`uv sync --frozen`).
- Estructura: `src/{config,etl,features,training,evaluation,inference,utils}`,
  `notebooks/` (EDA emparejado con `jupytext`), `data/raw/` (no se versiona).
- Ejecutar módulos con `PYTHONPATH=src uv run python -m <modulo>`.

## Políticas Compartidas

### Idioma

- **Todo el repositorio debe estar en español de México**: instrucciones,
  documentación, README, comentarios, docstrings, mensajes de `logging`,
  mensajes de excepciones, mensajes de CLI, nombres descriptivos de pruebas y
  mensajes de commit.
- Todo contenido nuevo o modificado debe redactarse en español de México; si se
  encuentra contenido en otro idioma, debe traducirse.
- Solo se conserva el idioma original en: identificadores de código (variables,
  funciones, clases, módulos), claves de JSON/YAML/frontmatter, nombres de
  archivos, comandos, nombres propios y de productos, estándares técnicos y
  literales que un script o herramienta interprete.
- Si un término técnico no tiene traducción de uso común (por ejemplo,
  `pipeline`, `commit`, `feature`), se puede conservar tal cual.

### Reglas Obligatorias

1. Todo desarrollo debe seguir CRISP-DM (entendimiento del negocio y de los
   datos, preparación, modelado, evaluación, despliegue).
2. Todo entregable debe tener evidencia verificable en GitHub (README,
   documentación, PRs, commits, logs o pruebas).
3. No se aceptan puntos sin evidencia verificable.
4. No se deben exponer secretos, tokens, credenciales ni datos sensibles. El
   token de Kaggle vive en `~/.kaggle/access_token`, nunca en el repositorio.
5. Los datos crudos (`data/raw/`) no se versionan.
6. Todo código debe incluir logging, manejo de errores, docstrings y pruebas
   cuando aplique.
7. Toda transformación de datos debe definir reglas de calidad de datos (DQ) y
   cifras de control (filas de entrada y salida, nulos, duplicados).
8. Los experimentos deben ser reproducibles: semillas fijas, dependencias
   bloqueadas en `uv.lock` y configuración externa.
9. Evitar fuga de datos (*data leakage*): el ajuste de transformaciones se hace
   solo con datos de entrenamiento y la validación debe reflejar el uso real.
10. Las decisiones de arquitectura o de modelado relevantes se documentan
    (ADR en `docs/adr/` cuando exista).

### Arquitectura y Principios de Ingeniería

- Preferir legibilidad, implementaciones explícitas y abstracciones
  justificadas.
- Mentalidad production-first: logging, manejo de errores, testabilidad,
  observabilidad y configuración externa.
- Separar lógica de negocio, infraestructura y acceso a datos.
- Mantener workflows reproducibles en local, Docker y CI/CD.

### Estándares de Código

- Python 3.11+ compatible.
- PEP 8, compatible con Black y líneas de máximo 88 caracteres.
- Módulos y funciones en `snake_case`, clases en `PascalCase`, constantes en
  `UPPER_SNAKE_CASE`.
- Todos los parámetros y retornos deben tener anotaciones de tipo.
- Docstrings estilo Google en funciones y clases públicas.
- Usar `logging`, nunca `print()`, para observabilidad de aplicación.
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

### Quality Gates

- [ ] PEP 8 compliance
- [ ] Type annotations
- [ ] Docstrings (Google style)
- [ ] Error handling y logging
- [ ] Sin hardcoding de credenciales
