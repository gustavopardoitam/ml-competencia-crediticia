"""Submissions para Kaggle: entrenar con el 100% de train y predecir test."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = ROOT / "data" / "processed"


def make_submission(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    ids: pd.Series,
    sample: pd.DataFrame,
    filename: str,
    out_dir: Path = SUBMISSION_DIR,
) -> pd.DataFrame:
    """Entrena `model` con todo train, predice test y guarda el CSV con el formato de `sample`."""
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"Id": ids.to_numpy(), "SeriousDlqin2yrs": proba})

    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    assert (submission["Id"].to_numpy() == sample["Id"].to_numpy()).all()

    out_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_dir / filename, index=False)
    print(f"{filename}: guardado, rango {proba.min():.4f}-{proba.max():.4f}")
    return submission
