import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from inference.submission import make_submission


def _data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, size=n))
    X = pd.DataFrame({"x": y + rng.normal(0, 0.5, size=n)})
    return X, y


def test_make_submission_formato(tmp_path):
    X, y = _data()
    X_test, _ = _data(n=20, seed=1)
    ids = pd.Series(range(1000, 1020))
    sample = pd.DataFrame({"Id": ids, "SeriousDlqin2yrs": 0.5})
    sub = make_submission(LogisticRegression(), X, y, X_test, ids, sample, "s.csv",
                          out_dir=tmp_path)
    saved = pd.read_csv(tmp_path / "s.csv")
    assert list(saved.columns) == ["Id", "SeriousDlqin2yrs"]
    assert (saved["Id"] == ids).all() and saved["SeriousDlqin2yrs"].between(0, 1).all()
    pd.testing.assert_frame_equal(saved, sub)


def test_make_submission_falla_si_los_ids_no_coinciden(tmp_path):
    X, y = _data()
    X_test, _ = _data(n=20, seed=1)
    sample = pd.DataFrame({"Id": range(20), "SeriousDlqin2yrs": 0.5})
    with pytest.raises(AssertionError):
        make_submission(LogisticRegression(), X, y, X_test, pd.Series(range(1, 21)), sample,
                        "s.csv", out_dir=tmp_path)
