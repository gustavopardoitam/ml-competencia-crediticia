"""Descarga los datos de la competencia a data/raw/ (no se versionan)."""
import shutil
from pathlib import Path

import kagglehub

COMPETITION = "competencia-crediticia-por-5-decimas"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def main() -> None:
    cache_path = Path(kagglehub.competition_download(COMPETITION))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in cache_path.rglob("*.csv"):
        shutil.copy2(f, RAW_DIR / f.name)
        print(f"copiado: {f.name}")
    print("Archivos en data/raw:", sorted(p.name for p in RAW_DIR.iterdir()))


if __name__ == "__main__":
    main()
