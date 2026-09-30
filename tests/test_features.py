import numpy as np
import pandas as pd
import pytest

from credit_scoring.features import build_features, clean


@pytest.fixture
def raw():
    return pd.DataFrame({
        "default": [0, 1, 0, 0],
        "revolving_util": [0.1, 1.2, 0.5, 0.3],
        "age": [30, 45, 0, 60],
        "late_30_59": [0, 98, 1, 2],
        "late_60_89": [0, 98, 0, 1],
        "late_90": [0, 98, 0, 0],
        "debt_ratio": [0.3, 0.5, 1500.0, 800.0],
        "monthly_income": [5000.0, 3000.0, np.nan, 0.0],
        "open_credit_lines": [5, 3, 8, 2],
        "real_estate_loans": [1, 0, 2, 0],
        "dependents": [2.0, 0.0, np.nan, 1.0],
    })


def test_special_late_codes_become_nan_with_flag(raw):
    out = clean(raw)
    assert out.loc[1, ["late_30_59", "late_60_89", "late_90"]].isna().all()
    assert out["late_special_code"].tolist() == [0, 1, 0, 0]


def test_missing_or_zero_income_moves_debt_ratio_to_monthly_debt(raw):
    out = clean(raw)
    assert out["income_missing"].tolist() == [0, 0, 1, 1]
    assert out.loc[[2, 3], ["monthly_income", "debt_ratio"]].isna().all().all()
    assert out["monthly_debt"].tolist() == pytest.approx([1500.0, 1500.0, 1500.0, 800.0])


def test_impossible_age_becomes_nan(raw):
    assert np.isnan(clean(raw).loc[2, "age"])


def test_derived_features(raw):
    out = build_features(raw)
    assert out.loc[3, "total_late"] == 3
    assert np.isnan(out.loc[1, "total_late"])  # códigos especiais não somam
    assert out.loc[0, "income_per_person"] == pytest.approx(5000 / 3)


def test_does_not_mutate_input(raw):
    before = raw.copy()
    build_features(raw)
    pd.testing.assert_frame_equal(raw, before)
