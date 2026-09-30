import numpy as np
import pandas as pd

from credit_scoring.gbm import MONOTONE, fit, make_params
from credit_scoring.metrics import calibration_table

rng = np.random.default_rng(0)


def test_monotone_vector_follows_feature_order():
    params = make_params(["revolving_util", "late_90", "age"])
    assert params["monotone_constraints"] == [0, MONOTONE["late_90"], MONOTONE["age"]]
    assert "monotone_constraints" not in make_params(["late_90"], monotone=False)


def test_monotone_constraint_is_enforced():
    # dados com relação não monotônica: o modelo restrito não pode reproduzi-la
    n = 4000
    X = pd.DataFrame({"late_90": rng.integers(0, 6, n).astype(float), "noise": rng.normal(size=n)})
    logit = -2 + np.where(X["late_90"] == 3, -2.0, 0.5 * X["late_90"])
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))))
    booster = fit(make_params(list(X.columns), num_leaves=7, min_child_samples=20), X, y, rounds=200)
    grid = pd.DataFrame({"late_90": np.arange(6.0), "noise": 0.0})
    assert np.all(np.diff(booster.predict(grid)) >= 0)


def test_calibration_table_of_perfect_probabilities():
    p = rng.uniform(0, 0.3, 50_000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p)
    np.testing.assert_allclose(table["prevista"], table["observada"], atol=0.01)
