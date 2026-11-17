import numpy as np
import pandas as pd

from features.features import prepare, prepare_min


def _row(**kw):
    base = {
        "Id": 1, "SeriousDlqin2yrs": 0,
        "RevolvingUtilizationOfUnsecuredLines": 0.5, "age": 40,
        "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": 0.3, "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 5, "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1, "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    base.update(kw)
    return base


def test_sentinel_and_age():
    out = prepare(pd.DataFrame([_row(NumberOfTimes90DaysLate=98, age=0)]))
    assert out["late_sentinel"].iloc[0] == 1
    assert np.isnan(out["NumberOfTimes90DaysLate"].iloc[0])
    assert np.isnan(out["age"].iloc[0])


def test_debt_amount_rules():
    out = prepare(pd.DataFrame([
        _row(MonthlyIncome=np.nan, DebtRatio=1200.0),
        _row(MonthlyIncome=4000.0, DebtRatio=0.5),
    ]))
    assert np.isnan(out["debt_ratio_valid"].iloc[0]) and out["debt_amount"].iloc[0] == 1200.0
    assert out["debt_ratio_valid"].iloc[1] == 0.5 and out["debt_amount"].iloc[1] == 2000.0


def test_no_target_or_id_columns():
    out = prepare(pd.DataFrame([_row()]))
    assert "SeriousDlqin2yrs" not in out.columns and "Id" not in out.columns


def test_row_independence_no_leakage():
    df = pd.DataFrame([_row(), _row(age=70, MonthlyIncome=np.nan, DebtRatio=900.0)])
    full = prepare(df)
    single = prepare(df.iloc[[1]])
    pd.testing.assert_frame_equal(full.iloc[[1]], single)


def test_prepare_min_columns():
    out = prepare_min(pd.DataFrame([_row()]))
    assert out.shape[1] == 16
    assert not {"util_clip", "debt_amount", "late_x_util"} & set(out.columns)
