import numpy as np
import pytest

from credit_scoring.cutoff import ProfitAssumptions, optimal, profit_curve, simulate

rng = np.random.default_rng(0)


def test_assumption_arithmetic():
    a = ProfitAssumptions(ticket=10_000, margin_rate=0.15, ead_rate=0.8, lgd=0.6)
    assert a.gain_good == pytest.approx(1_500)
    assert a.loss_bad == pytest.approx(4_800)
    assert a.breakeven_pd == pytest.approx(1_500 / 6_300)


def test_simulate_counts_and_profit():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.5, 0.2, 0.9])
    a = ProfitAssumptions(ticket=100, margin_rate=0.1, ead_rate=1.0, lgd=1.0)
    r = simulate(y, p, threshold=0.3, assumptions=a)
    assert (r["bons_aprovados"], r["maus_aprovados"], r["bons_recusados"]) == (1, 1, 1)
    assert r["lucro"] == pytest.approx(10 - 100)
    assert r["aprovação"] == pytest.approx(0.5)


def test_optimum_is_breakeven_when_probabilities_are_calibrated():
    p = rng.uniform(0, 0.6, 400_000)
    y = rng.binomial(1, p)
    a = ProfitAssumptions()
    best = optimal(profit_curve(y, p, a))
    assert best["cutoff"] == pytest.approx(a.breakeven_pd, abs=0.015)
