import numpy as np
import pandas as pd

from credit_scoring.woe import MISSING, WoEEncoder, auto_edges, woe_table

rng = np.random.default_rng(0)


def test_auto_edges_never_splits_a_value_and_respects_min_size():
    x = pd.Series([0] * 850 + [1] * 100 + [2] * 30 + [5] * 20)
    edges = auto_edges(x, n_bins=10, min_bin_frac=0.05)
    assert edges == [0.0, 1.0]  # 0 | 1 | 2+ (o 5 sozinho seria pequeno demais)


def test_auto_edges_keeps_rare_binary_flag():
    x = pd.Series([0] * 998 + [1] * 2)
    assert auto_edges(x) == [0.0]


def test_auto_edges_continuous_gives_roughly_equal_bins():
    x = pd.Series(rng.normal(size=10_000))
    edges = auto_edges(x, n_bins=10)
    counts = pd.cut(x, [-np.inf, *edges, np.inf]).value_counts()
    assert len(counts) == 10
    assert counts.min() > 900


def test_iv_high_for_informative_feature_and_near_zero_for_noise():
    n = 20_000
    y = pd.Series(rng.binomial(1, 0.1, n))
    informative = pd.Series(y * 2 + rng.normal(size=n))
    noise = pd.Series(rng.normal(size=n))
    iv_inf = woe_table(informative, y, auto_edges(informative))["iv"].sum()
    iv_noise = woe_table(noise, y, auto_edges(noise))["iv"].sum()
    assert iv_inf > 0.5
    assert iv_noise < 0.02


def test_missing_gets_its_own_bin_and_woe_sign():
    # faltantes são todos maus -> WoE negativo (mais risco que a média)
    x = pd.Series([1.0] * 900 + [np.nan] * 100)
    y = pd.Series([0] * 900 + [1] * 100)
    table = woe_table(x, y, edges=[]).set_index("bin")
    assert table.loc[MISSING, "woe"] < 0
    assert table.loc["todos", "woe"] > 0


def test_tiny_bin_gets_neutral_woe():
    x = pd.Series([1.0] * 999 + [np.nan])
    y = pd.Series([0] * 900 + [1] * 99 + [0])
    table = woe_table(x, y, edges=[]).set_index("bin")
    assert table.loc[MISSING, "woe"] == 0.0
    assert table.loc[MISSING, "iv"] == 0.0


def test_encoder_transform_and_unseen_missing_is_neutral():
    X = pd.DataFrame({"a": rng.normal(size=1000)})
    y = pd.Series(rng.binomial(1, 0.2, 1000))
    enc = WoEEncoder().fit(X, y)
    out = enc.transform(pd.DataFrame({"a": [0.0, np.nan]}))
    assert out.shape == (2, 1)
    assert out.loc[1, "a"] == 0.0  # não havia faltante no treino
    assert "a" in enc.iv_.index
