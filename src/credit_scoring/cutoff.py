"""Simulação financeira da política de crédito e escolha do ponto de corte.

Regra de decisão: aprova a proposta se a probabilidade de default prevista for
menor ou igual ao cutoff.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfitAssumptions:
    """Premissas de um empréstimo típico. Valores ilustrativos, fáceis de trocar.

    ticket: valor emprestado (R$)
    margin_rate: lucro líquido de um bom pagador ao longo do contrato, em % do ticket
        (juros menos custo de captação, operacional e impostos)
    ead_rate: parte do ticket ainda em aberto quando o default acontece (exposure at default)
    lgd: parte da exposição que não é recuperada (loss given default)
    """

    ticket: float = 10_000.0
    margin_rate: float = 0.15
    ead_rate: float = 0.80
    lgd: float = 0.60

    @property
    def gain_good(self) -> float:
        return self.ticket * self.margin_rate

    @property
    def loss_bad(self) -> float:
        return self.ticket * self.ead_rate * self.lgd

    @property
    def breakeven_pd(self) -> float:
        """PD em que o lucro esperado de aprovar é zero: (1 - p)·ganho = p·perda."""
        return self.gain_good / (self.gain_good + self.loss_bad)


DEFAULT_THRESHOLDS = np.round(np.arange(0.005, 0.605, 0.005), 3)


def simulate(y_true, p_default, threshold: float, assumptions: ProfitAssumptions) -> dict:
    y = np.asarray(y_true)
    approved = np.asarray(p_default) <= threshold
    goods = int(((y == 0) & approved).sum())
    bads = int(((y == 1) & approved).sum())
    revenue = goods * assumptions.gain_good
    losses = bads * assumptions.loss_bad
    return {
        "cutoff": float(threshold),
        "aprovação": approved.mean(),
        "aprovados": int(approved.sum()),
        "bons_aprovados": goods,
        "maus_aprovados": bads,
        "inadimplência_carteira": bads / max(goods + bads, 1),
        "bons_recusados": int(((y == 0) & ~approved).sum()),
        "margem": revenue,
        "perdas": losses,
        "lucro": revenue - losses,
    }


def profit_curve(y_true, p_default, assumptions: ProfitAssumptions, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    return pd.DataFrame([simulate(y_true, p_default, t, assumptions) for t in thresholds])


def optimal(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["lucro"].idxmax()]
