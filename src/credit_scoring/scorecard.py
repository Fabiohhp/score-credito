"""Scorecard de pontos a partir de uma regressão logística sobre WoE.

Escala usual de mercado: um score base (600) corresponde a uma chance de
bom:mau (50:1), e a cada PDO pontos (20) essa chance dobra.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scoring.woe import WoEEncoder

# Faixas agrupadas à mão (coarse classing) no notebook 03. Limites superiores, inclusivos.
COARSE_EDGES = {
    "revolving_util": [0, 0.05, 0.13, 0.23, 0.39, 0.5, 0.7, 0.9, 1.0],
    "age": [33, 40, 48, 53, 56, 62, 69],
    "debt_ratio": [0.4256, 0.5246, 0.7127, 0.9491],
    "late_30_59": [0, 1, 2],
    "late_60_89": [0, 1],
    "late_90": [0, 1],
    "real_estate_loans": [0, 2],
}
SCORECARD_FEATURES = list(COARSE_EDGES)

BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20


def make_logistic_pipeline(edges: dict | None = None) -> Pipeline:
    return Pipeline([
        ("woe", WoEEncoder(edges=COARSE_EDGES if edges is None else edges)),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])


def _scaling(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO) -> tuple[float, float]:
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def proba_to_score(p_bad, **scaling) -> np.ndarray:
    """Score contínuo equivalente à probabilidade de default."""
    factor, offset = _scaling(**scaling)
    p_bad = np.clip(np.asarray(p_bad, dtype=float), 1e-12, 1 - 1e-12)
    return offset + factor * np.log((1 - p_bad) / p_bad)


def point_params(pipeline: Pipeline, **scaling) -> tuple[float, pd.Series]:
    """Pontos base por variável e o peso (pontos por unidade de WoE) de cada uma.

    O modelo prevê log(chance de mau) = b0 + Σ βi·WoEi. Como o WoE é
    ln(%bons/%maus), os β saem negativos e o score, que mede chance de bom, é
    score = offset - factor·(b0 + Σ βi·WoEi). O intercepto é dividido igualmente
    entre as variáveis para que cada faixa carregue um número de pontos completo:
    pontos_i = base_points - factor·βi·WoEi.
    """
    encoder: WoEEncoder = pipeline.named_steps["woe"]
    model: LogisticRegression = pipeline.named_steps["logreg"]
    factor, offset = _scaling(**scaling)
    base_points = (offset - factor * model.intercept_[0]) / len(encoder.feature_names_in_)
    weights = pd.Series(-factor * model.coef_[0], index=encoder.feature_names_in_)
    return base_points, weights


def build_scorecard(pipeline: Pipeline, rounded: bool = True, **scaling) -> pd.DataFrame:
    """Tabela de pontos por faixa de cada variável."""
    encoder: WoEEncoder = pipeline.named_steps["woe"]
    base_points, weights = point_params(pipeline, **scaling)
    tables = []
    for feature, table in encoder.tables_.items():
        tables.append(pd.DataFrame({
            "variável": feature,
            "faixa": table["bin"].astype(str),
            "n_treino": table["n"],
            "taxa_default": table["bad_rate"],
            "woe": table["woe"],
            "pontos": base_points + weights[feature] * table["woe"],
        }))
    scorecard = pd.concat(tables, ignore_index=True)
    if rounded:
        scorecard["pontos"] = scorecard["pontos"].round().astype(int)
    return scorecard


def score_points(pipeline: Pipeline, X: pd.DataFrame, rounded: bool = True, **scaling) -> pd.DataFrame:
    """Pontos de cada variável para cada proposta (colunas = variáveis).

    Faixas que não existiam no treino têm WoE 0 e recebem os pontos de risco médio.
    """
    woe = pipeline.named_steps["woe"].transform(X)
    base_points, weights = point_params(pipeline, **scaling)
    points = base_points + woe * weights
    return points.round().astype(int) if rounded else points


def score(pipeline: Pipeline, X: pd.DataFrame, rounded: bool = True, **scaling) -> pd.Series:
    return score_points(pipeline, X, rounded, **scaling).sum(axis=1).rename("score")
