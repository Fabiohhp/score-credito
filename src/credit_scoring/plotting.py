"""Estilo visual comum aos notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns

from credit_scoring.data import ROOT

FIG_DIR = ROOT / "reports" / "figures"

# azul = adimplente / geral, laranja = inadimplente / destaque de risco
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"


def set_style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "#d0cfca", "grid.color": "#ecebe7", "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
        "figure.dpi": 110, "savefig.bbox": "tight",
    })


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
