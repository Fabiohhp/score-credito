"""LightGBM com restrições de monotonicidade e escolha de hiperparâmetros por validação cruzada."""
import itertools

import lightgbm as lgb
import pandas as pd

from credit_scoring.data import RANDOM_STATE

# Direção obrigatória do efeito sobre o risco: +1 = só pode aumentar, -1 = só pode diminuir.
# Variáveis de fora (ex.: revolving_util, com o "zero" mais arriscado que "pouco uso",
# e as de forma de U) ficam livres.
MONOTONE = {
    "late_30_59": 1,
    "late_60_89": 1,
    "late_90": 1,
    "total_late": 1,
    "late_special_code": 1,
    "debt_ratio": 1,
    "age": -1,
    "monthly_income": -1,
    "income_per_person": -1,
}

BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": RANDOM_STATE,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def make_params(features: list[str], monotone: bool = True, **overrides) -> dict:
    params = {**BASE_PARAMS, **overrides}
    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(f, 0) for f in features]
        params["monotone_constraints_method"] = "advanced"
    return params


def cv_best_rounds(params: dict, X: pd.DataFrame, y: pd.Series, nfold: int = 5) -> tuple[int, float]:
    """Número de árvores escolhido por early stopping numa validação cruzada estratificada."""
    result = lgb.cv(
        params, lgb.Dataset(X, y), num_boost_round=5000, nfold=nfold, stratified=True,
        seed=RANDOM_STATE, callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    auc = result["valid auc-mean"]
    return len(auc), float(auc[-1])


def grid_search(X: pd.DataFrame, y: pd.Series, grid: dict[str, list], monotone_options=(False, True)) -> pd.DataFrame:
    rows = []
    for monotone, values in itertools.product(monotone_options, itertools.product(*grid.values())):
        overrides = dict(zip(grid, values))
        rounds, auc = cv_best_rounds(make_params(list(X.columns), monotone, **overrides), X, y)
        rows.append({"monotone": monotone, **overrides, "n_arvores": rounds, "auc_cv": auc})
    return pd.DataFrame(rows).sort_values("auc_cv", ascending=False, ignore_index=True)


def fit(params: dict, X: pd.DataFrame, y: pd.Series, rounds: int) -> lgb.Booster:
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)
