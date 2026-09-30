"""Métricas de crédito: ROC-AUC, Gini e KS."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Distância máxima entre as distribuições acumuladas de maus e bons."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y_true, y_score)}


def band_table(y_true, score, n_bands: int = 10) -> pd.DataFrame:
    """Taxa de default por faixa de score (faixa 1 = scores mais baixos = mais risco)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(score)})
    df["faixa"] = pd.qcut(df["score"].rank(method="first"), n_bands, labels=range(1, n_bands + 1))
    table = df.groupby("faixa", observed=True).agg(
        score_min=("score", "min"), score_max=("score", "max"), n=("y", "size"), maus=("y", "sum")
    )
    table["taxa_default"] = table["maus"] / table["n"]
    table["%_maus_acumulado"] = table["maus"].cumsum() / table["maus"].sum()
    return table


def calibration_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    """Probabilidade média prevista vs taxa observada, em faixas de mesma quantidade."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    df["faixa"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    return df.groupby("faixa", observed=True).agg(prevista=("p", "mean"), observada=("y", "mean"), n=("y", "size"))
