# ML Championship: competencia crediticia

Predicción de P(SeriousDlqin2yrs = 1). Métrica: AUC-ROC.

## Setup
`uv sync --frozen`. Los datos van en `data/raw/` y no se versionan.

## Reproducir el EDA
1. `git clone` y `uv sync`.
2. Acepta las reglas de la competencia en Kaggle y guarda tu token en `~/.kaggle/access_token` (nunca en el repo).
3. `PYTHONPATH=src uv run python -m etl.download`
4. Abre `notebooks/01_eda.ipynb` con el kernel `.venv`. El `.py` emparejado sirve para revisar diffs; sincroniza con `uv run jupytext --sync notebooks/01_eda.py`.
