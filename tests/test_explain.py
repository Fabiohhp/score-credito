import numpy as np
import pandas as pd
import pytest

from credit_scoring.explain import (
    GROUP_LABELS, REASON_GROUPS, describe, group_contributions, lgbm_reason_codes, shap_values, top_reasons,
)
from credit_scoring.features import FEATURES
from credit_scoring.gbm import fit, make_params

rng = np.random.default_rng(0)


def _row(**overrides):
    base = {"late_30_59": 0, "late_60_89": 0, "late_90": 0, "late_special_code": 0, "revolving_util": 0.5,
            "debt_ratio": 0.3, "monthly_debt": 1500.0, "monthly_income": 5000.0, "dependents": 1.0,
            "age": 40, "open_credit_lines": 5, "real_estate_loans": 1}
    return pd.Series({**base, **overrides})


def test_every_feature_has_a_reason_group():
    assert set(FEATURES) <= set(REASON_GROUPS)
    assert set(REASON_GROUPS.values()) == set(GROUP_LABELS)


@pytest.mark.parametrize("group, overrides, expected", [
    ("atrasos", {"late_90": 1, "late_30_59": 2}, "1 atraso de 90 dias ou mais, 2 atrasos de 30 a 59 dias"),
    ("atrasos", {"late_special_code": 1}, "sem detalhamento"),
    ("uso_rotativo", {"revolving_util": 1.3}, "acima do limite contratado (130%"),
    ("uso_rotativo", {"revolving_util": 0.0}, "Nenhum uso recente"),
    ("endividamento", {"debt_ratio": np.nan, "monthly_debt": 2500.0}, "R$ 2.500 sem renda informada"),
    ("renda", {"monthly_income": np.nan}, "não informada"),
    ("renda", {"dependents": np.nan}, "para 1 pessoa"),
])
def test_describe(group, overrides, expected):
    assert expected in describe(group, _row(**overrides))


def test_group_contributions_sum_within_groups():
    contrib = pd.DataFrame({"late_90": [1.0], "total_late": [0.5], "age": [-0.2]})
    out = group_contributions(contrib)
    assert out.loc[0, "atrasos"] == pytest.approx(1.5)
    assert out.loc[0, "idade"] == pytest.approx(-0.2)


def test_top_reasons_ignores_factors_that_reduced_risk():
    impact = pd.DataFrame({"atrasos": [30.0], "idade": [-5.0], "uso_rotativo": [10.0]})
    X = pd.DataFrame([_row(late_90=2)])
    out = top_reasons(impact, X, n=3)
    assert out.loc[0, "motivo_1"] == GROUP_LABELS["atrasos"]
    assert out.loc[0, "motivo_2"] == GROUP_LABELS["uso_rotativo"]
    assert "motivo_3" not in out.columns


def test_shap_is_additive_and_reasons_point_to_the_driver():
    n = 3000
    X = pd.DataFrame({"late_90": rng.integers(0, 4, n).astype(float), "age": rng.integers(20, 80, n).astype(float)})
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(-3 + 1.5 * X["late_90"])))))
    booster = fit(make_params(list(X.columns), num_leaves=7, min_child_samples=20), X, y, rounds=100)
    contrib, base = shap_values(booster, X)
    np.testing.assert_allclose(contrib.sum(axis=1) + base, booster.predict(X, raw_score=True), atol=1e-6)

    # colunas extras (usadas só no texto do motivo) não podem atrapalhar o modelo
    risky = X[X["late_90"] == 3].head(5).assign(late_30_59=0, late_60_89=0)
    reasons = lgbm_reason_codes(booster, risky)
    assert (reasons["motivo_1"] == GROUP_LABELS["atrasos"]).all()
