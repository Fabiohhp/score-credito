"""Métricas salvas em reports/metrics.json: fonte única dos números citados no README."""
import json

from credit_scoring.data import ROOT

METRICS_PATH = ROOT / "reports" / "metrics.json"


def update_metrics(section: str, values: dict) -> dict:
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics[section] = values
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def brl(value: float, decimals: int = 0) -> str:
    """Formata em reais no padrão brasileiro: R$ 1.234.567."""
    text = f"{abs(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{'-' if value < 0 else ''}R$ {text}"
