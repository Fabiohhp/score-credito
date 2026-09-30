"""Carga dos dados brutos e divisão treino/validação/teste."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "default"
RANDOM_STATE = 42

RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}
LATE_COLS = ["late_30_59", "late_60_89", "late_90"]


def load_raw(name: str = "cs-training.csv") -> pd.DataFrame:
    """Lê um CSV do Kaggle e renomeia as colunas para snake_case."""
    df = pd.read_csv(RAW_DIR / name, index_col=0).rename(columns=RENAME)
    df.index.name = "id"
    return df


def split_train_val_test(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divisão estratificada pelo alvo em treino, validação e teste."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET],
        random_state=RANDOM_STATE,
    )
    return train, val, test
