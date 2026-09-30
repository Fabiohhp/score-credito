"""Binning, Weight of Evidence (WoE) e Information Value (IV).

Convenção: WoE = ln(%bons / %maus). WoE positivo = faixa com menos risco que a média.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING = "Faltante"

# Faixas de referência usuais do mercado para interpretar o IV
IV_LABELS = [
    (0.02, "sem poder"),
    (0.10, "fraco"),
    (0.30, "médio"),
    (0.50, "forte"),
    (np.inf, "muito forte (checar vazamento)"),
]


def iv_strength(iv: float) -> str:
    return next(label for limit, label in IV_LABELS if iv < limit)


def auto_edges(x: pd.Series, n_bins: int = 10, min_bin_frac: float = 0.01) -> list[float]:
    """Limites superiores de faixas com frequência aproximadamente igual.

    Percorre os valores únicos em ordem e fecha uma faixa quando ela atinge
    ~1/n_bins da base. Funciona para contínuas e contagens (um valor nunca é
    dividido entre duas faixas). A última faixa é unida à anterior se ficar
    menor que `min_bin_frac`. Variáveis binárias nunca são unidas.
    """
    x = x.dropna()
    values, counts = np.unique(x, return_counts=True)
    if len(values) <= 2:  # binária: cada valor é uma faixa, por menor que seja
        return [float(values[0])] if len(values) == 2 else []
    target, min_n = len(x) / n_bins, min_bin_frac * len(x)

    edges, acc = [], 0
    for value, count in zip(values, counts):
        acc += count
        if acc >= target:
            edges.append(float(value))
            acc = 0
    if edges and edges[-1] == values[-1]:
        edges.pop()
        acc = counts[values > (edges[-1] if edges else -np.inf)].sum()
    if edges and acc < min_n:
        edges.pop()
    return edges


def _fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _bin_labels(edges: list[float]) -> list[str]:
    bounds = [-np.inf, *edges, np.inf]
    labels = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo == -np.inf and hi == np.inf:
            labels.append("todos")
        elif lo == -np.inf:
            labels.append(f"<= {_fmt(hi)}")
        elif hi == np.inf:
            labels.append(f"> {_fmt(lo)}")
        else:
            labels.append(f"({_fmt(lo)}, {_fmt(hi)}]")
    return labels


def assign_bins(x: pd.Series, edges: list[float]) -> pd.Series:
    labels = _bin_labels(edges)
    binned = pd.cut(x, bins=[-np.inf, *edges, np.inf], labels=labels)
    return binned.cat.add_categories(MISSING).fillna(MISSING)


def woe_table(
    x: pd.Series, y: pd.Series, edges: list[float], smoothing: float = 0.5, min_count: int = 50
) -> pd.DataFrame:
    """Tabela de WoE/IV por faixa.

    `smoothing` evita log(0) em faixas sem bons ou sem maus. Faixas com menos de
    `min_count` observações (ex.: o faltante de uma variável quase sempre
    preenchida) não têm evidência suficiente e recebem WoE 0, isto é, risco médio.
    """
    bins = assign_bins(x, edges)
    table = (
        pd.DataFrame({"bin": bins, "bad": y})
        .groupby("bin", observed=True)["bad"]
        .agg(n="size", bad="sum")
    )
    table["good"] = table["n"] - table["bad"]
    table["bad_rate"] = table["bad"] / table["n"]
    dist_good = (table["good"] + smoothing) / (table["good"].sum() + smoothing * len(table))
    dist_bad = (table["bad"] + smoothing) / (table["bad"].sum() + smoothing * len(table))
    table["woe"] = np.log(dist_good / dist_bad).where(table["n"] >= min_count, 0.0)
    table["iv"] = (dist_good - dist_bad) * table["woe"]
    return table.reset_index()


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Substitui cada variável pelo WoE da faixa em que o valor cai.

    `edges` permite passar faixas definidas manualmente (coarse classing) por
    variável; as demais usam `auto_edges`.
    """

    def __init__(self, n_bins: int = 10, min_bin_frac: float = 0.01, edges: dict | None = None):
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        self.edges = edges

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)
        manual = self.edges or {}
        self.edges_, self.tables_ = {}, {}
        for col in X.columns:
            edges = manual[col] if col in manual else auto_edges(X[col], self.n_bins, self.min_bin_frac)
            self.edges_[col] = edges
            self.tables_[col] = woe_table(X[col], y, edges)
        self.iv_ = (
            pd.Series({col: t["iv"].sum() for col, t in self.tables_.items()}, name="iv")
            .sort_values(ascending=False)
        )
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.feature_names_in_:
            mapping = self.tables_[col].set_index("bin")["woe"]
            bins = assign_bins(X[col], self.edges_[col]).astype(str)
            # Faixa sem observações no treino (ex.: faltante que não existia) recebe WoE 0 = neutro
            out[col] = bins.map(mapping.rename(index=str)).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
