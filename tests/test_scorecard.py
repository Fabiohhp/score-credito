import numpy as np
import pandas as pd
import pytest

from credit_scoring.metrics import band_table, evaluate, ks_statistic
from credit_scoring.scorecard import (
    build_scorecard, make_logistic_pipeline, proba_to_score, score, score_points,
)
from credit_scoring.woe import assign_bins

rng = np.random.default_rng(0)


@pytest.fixture(scope="module")
def fitted():
    n = 5000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, n).astype(float)})
    logit = -2.5 + 1.2 * X["a"] + 0.4 * X["b"]
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))))
    pipe = make_logistic_pipeline(edges={"a": [-1, 0, 1], "b": [0, 1, 2, 3]}).fit(X, y)
    return pipe, X, y


def test_unrounded_score_matches_probability(fitted):
    pipe, X, _ = fitted
    expected = proba_to_score(pipe.predict_proba(X)[:, 1])
    np.testing.assert_allclose(score(pipe, X, rounded=False), expected, rtol=1e-9)


def test_scorecard_table_matches_points_per_applicant(fitted):
    pipe, X, _ = fitted
    card = build_scorecard(pipe)
    points = score_points(pipe, X.iloc[:1])
    encoder = pipe.named_steps["woe"]
    for feature in X.columns:
        bin_label = str(assign_bins(X[feature].iloc[:1], encoder.edges_[feature]).iloc[0])
        expected = card.loc[(card["variável"] == feature) & (card["faixa"] == bin_label), "pontos"].item()
        assert points[feature].iloc[0] == expected


def test_higher_score_means_lower_risk(fitted):
    pipe, X, y = fitted
    s = score(pipe, X)
    assert s[y == 0].mean() > s[y == 1].mean()


def test_scaling_doubles_odds_every_pdo():
    s1, s2 = proba_to_score([1 / 51, 1 / 101])  # chance bom:mau 50:1 e 100:1
    assert s1 == pytest.approx(600)
    assert s2 - s1 == pytest.approx(20)


def test_ks_extremes():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, y) == pytest.approx(1.0)
    assert evaluate(y, y)["gini"] == pytest.approx(1.0)
    assert ks_statistic(y, np.full(100, 0.5)) == pytest.approx(0.0)


def test_band_table_covers_everyone():
    y = rng.binomial(1, 0.1, 1000)
    table = band_table(y, rng.normal(size=1000), n_bands=10)
    assert table["n"].sum() == 1000
    assert table["%_maus_acumulado"].iloc[-1] == pytest.approx(1.0)
