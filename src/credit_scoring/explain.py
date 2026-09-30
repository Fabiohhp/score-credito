"""Explicações com SHAP e reason codes (motivos de recusa) em linguagem simples.

Variáveis que descrevem o mesmo aspecto do cliente (por exemplo, os três tipos de
atraso e o total de atrasos) são agrupadas num único motivo, para que a
explicação não repita a mesma coisa com nomes diferentes.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from credit_scoring.scorecard import PDO, point_params

REASON_GROUPS = {
    "late_30_59": "atrasos",
    "late_60_89": "atrasos",
    "late_90": "atrasos",
    "total_late": "atrasos",
    "late_special_code": "atrasos",
    "revolving_util": "uso_rotativo",
    "debt_ratio": "endividamento",
    "monthly_debt": "endividamento",
    "monthly_income": "renda",
    "income_per_person": "renda",
    "income_missing": "renda",
    "dependents": "renda",
    "age": "idade",
    "open_credit_lines": "relacionamento",
    "real_estate_loans": "relacionamento",
}

GROUP_LABELS = {
    "atrasos": "Histórico de atrasos",
    "uso_rotativo": "Uso do crédito rotativo",
    "endividamento": "Comprometimento com dívidas",
    "renda": "Renda",
    "idade": "Idade",
    "relacionamento": "Linhas de crédito e financiamentos",
}

# 1 unidade de log-odds equivale a PDO / ln(2) pontos de score
POINTS_PER_LOGODDS = PDO / np.log(2)


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Contribuição de cada variável para o log-odds de default, e o valor base."""
    X = X[booster.feature_name()]  # X pode trazer colunas extras, usadas só na descrição
    explainer = shap.TreeExplainer(booster)
    values, base = explainer.shap_values(X), explainer.expected_value
    if isinstance(values, list):  # algumas versões devolvem [classe 0, classe 1]
        values, base = values[-1], np.atleast_1d(base)[-1]
    return pd.DataFrame(values, index=X.index, columns=X.columns), float(np.ravel(base)[-1])


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    """Soma as contribuições das variáveis de cada grupo de motivo."""
    groups = contrib.columns.map(lambda c: REASON_GROUPS.get(c, c))
    return contrib.T.groupby(groups).sum().T


def _brl(value: float) -> str:
    return "R$ " + f"{value:,.0f}".replace(",", ".")


def _plural(n: float, one: str, many: str) -> str:
    return f"{n:.0f} {one if n == 1 else many}"


def describe(group: str, row: pd.Series) -> str:
    """Frase curta, em linguagem simples, com o dado do cliente por trás do motivo."""
    if group == "atrasos":
        if row.get("late_special_code", 0) == 1:
            return "Histórico de atrasos com registro sem detalhamento no bureau"
        parts = [
            _plural(row[col], f"atraso de {label}", f"atrasos de {label}")
            for col, label in [("late_90", "90 dias ou mais"), ("late_60_89", "60 a 89 dias"), ("late_30_59", "30 a 59 dias")]
            if row.get(col, 0) > 0
        ]
        return "Atrasos nos últimos 2 anos: " + ", ".join(parts) if parts else "Histórico de pagamentos recente"
    if group == "uso_rotativo":
        v = row["revolving_util"]
        if v == 0:
            return "Nenhum uso recente do crédito rotativo (pouco histórico para avaliar)"
        if v > 1:
            return f"Saldo do crédito rotativo acima do limite contratado ({v:.0%} do limite)"
        return f"Uso de {v:.0%} do limite do crédito rotativo"
    if group == "endividamento":
        if pd.isna(row.get("debt_ratio")):
            return f"Dívida mensal de {_brl(row['monthly_debt'])} sem renda informada para comparação"
        return f"Comprometimento de {row['debt_ratio']:.0%} da renda mensal com dívidas"
    if group == "renda":
        if pd.isna(row.get("monthly_income")):
            return "Renda mensal não informada"
        people = 1 + (0 if pd.isna(row.get("dependents")) else row["dependents"])
        return f"Renda mensal de {_brl(row['monthly_income'])} para {_plural(people, 'pessoa', 'pessoas')}"
    if group == "idade":
        return f"Faixa etária ({row['age']:.0f} anos) com maior risco histórico na carteira"
    if group == "relacionamento":
        return (f"{_plural(row['open_credit_lines'], 'linha de crédito aberta', 'linhas de crédito abertas')} e "
                f"{_plural(row['real_estate_loans'], 'financiamento imobiliário', 'financiamentos imobiliários')}")
    raise KeyError(group)


def top_reasons(group_impact: pd.DataFrame, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Os `n` grupos que mais aumentaram o risco de cada cliente.

    `group_impact` deve estar em "pontos de score perdidos" (positivo = aumentou o risco).
    Só entram grupos com impacto positivo: um fator que reduziu o risco nunca é motivo de recusa.
    """
    rows = []
    for idx, impact in group_impact.iterrows():
        top = impact[impact > 0].sort_values(ascending=False).head(n)
        row = {}
        for i, (group, points) in enumerate(top.items(), start=1):
            row[f"motivo_{i}"] = GROUP_LABELS[group]
            row[f"detalhe_{i}"] = describe(group, X.loc[idx])
            row[f"pontos_{i}"] = round(float(points))
        rows.append(row)
    out = pd.DataFrame(rows, index=group_impact.index)
    points_cols = [c for c in out.columns if c.startswith("pontos_")]
    return out.astype({c: "Int64" for c in points_cols})


def lgbm_reason_codes(booster: lgb.Booster, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    contrib, _ = shap_values(booster, X)
    # SHAP positivo aumenta o log-odds de default = tira pontos do score
    points_lost = group_contributions(contrib) * POINTS_PER_LOGODDS
    return top_reasons(points_lost, X, n)


def scorecard_reason_codes(pipeline: Pipeline, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Método clássico de scorecard: pontos perdidos em relação à melhor faixa de cada variável.

    Os pontos de uma faixa são base + peso·WoE, então o que o cliente deixou de
    ganhar numa variável é peso·(maior WoE da variável - WoE do cliente).
    """
    encoder = pipeline.named_steps["woe"]
    _, weights = point_params(pipeline)
    woe = encoder.transform(X)
    best_woe = pd.Series({f: encoder.tables_[f]["woe"].max() for f in woe.columns})
    points_lost = (best_woe - woe) * weights
    return top_reasons(group_contributions(points_lost), X, n)
