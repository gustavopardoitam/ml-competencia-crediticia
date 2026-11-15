"""Bitácora de experimentos: un registro por corrida de evaluate()."""
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LOG_PATH = Path(__file__).resolve().parents[2] / "reports" / "experiments.csv"
COLUMNS = ["timestamp", "nombre", "features", "modelo", "cv_auc_mean", "cv_auc_std", "kaggle_score", "notas"]


def log_experiment(
    nombre: str,
    features: str,
    modelo: str,
    cv_auc_mean: float,
    cv_auc_std: float,
    kaggle_score: float | None = None,
    notas: str = "",
    log_path: Path = LOG_PATH,
) -> pd.DataFrame:
    """Agrega una fila a la bitácora de experimentos y devuelve la bitácora completa."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "nombre": nombre,
        "features": features,
        "modelo": modelo,
        "cv_auc_mean": round(cv_auc_mean, 5),
        "cv_auc_std": round(cv_auc_std, 5),
        "kaggle_score": kaggle_score,
        "notas": notas,
    }
    df = read_log(log_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log_path, index=False)
    return df


def read_log(log_path: Path = LOG_PATH) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(log_path)
