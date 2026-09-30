"""Limpeza e criação de variáveis, seguindo os achados do notebook 01_eda."""
import numpy as np
import pandas as pd

from credit_scoring.data import LATE_COLS

# 96 e 98 nas colunas de atraso são códigos de sistema, não contagens
LATE_SPECIAL_CODES = (96, 98)
# Renda 0 ou 1 se comporta como renda não informada (debt_ratio vira valor absoluto)
MIN_VALID_INCOME = 2

FEATURES = [
    "revolving_util",
    "age",
    "late_30_59",
    "late_60_89",
    "late_90",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "real_estate_loans",
    "dependents",
    "income_missing",
    "late_special_code",
    "total_late",
    "monthly_debt",
    "income_per_person",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Converte códigos especiais e valores impossíveis em NaN e cria as flags."""
    out = df.copy()

    special = out[LATE_COLS].isin(LATE_SPECIAL_CODES).any(axis=1)
    out["late_special_code"] = special.astype(int)
    out[LATE_COLS] = out[LATE_COLS].mask(out[LATE_COLS].isin(LATE_SPECIAL_CODES))

    income_missing = out["monthly_income"].isna() | (out["monthly_income"] < MIN_VALID_INCOME)
    out["income_missing"] = income_missing.astype(int)
    # Sem renda válida, o debt_ratio guarda o valor da dívida mensal, não a razão
    out["monthly_debt"] = np.where(
        income_missing, out["debt_ratio"], out["debt_ratio"] * out["monthly_income"]
    )
    out.loc[income_missing, ["monthly_income", "debt_ratio"]] = np.nan

    out["age"] = out["age"].where(out["age"] >= 18)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variáveis derivadas. Espera um DataFrame já passado por `clean`."""
    out = df.copy()
    out["total_late"] = out[LATE_COLS].sum(axis=1, min_count=len(LATE_COLS))
    out["income_per_person"] = out["monthly_income"] / (out["dependents"].fillna(0) + 1)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(clean(df))
