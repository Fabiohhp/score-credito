"""Gera o README.md a partir de reports/metrics.json.

Nenhum número do README é digitado à mão: todos vêm das métricas salvas pelos
notebooks. Rode depois de executar os notebooks (scripts/run_all.sh faz isso).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "reports" / "metrics.json").read_text())


def num(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, decimals: int = 0) -> str:
    return num(100 * x, decimals) + "%"


def brl(x: float) -> str:
    return "R$ " + num(x)


def brl_mi(x: float) -> str:
    return "R$ " + num(x / 1e6, 2) + " mi"


def example_letter() -> str:
    reasons = pd.read_csv(ROOT / "reports" / "reason_codes_val.csv", index_col=0)
    # mesmo critério do notebook 06: primeira recusa em que o rotativo é o motivo principal
    row = reasons[(reasons["motivo_1"] == "Uso do crédito rotativo") & reasons["motivo_3"].notna()].iloc[0]
    lines = [f"{i}. **{row[f'motivo_{i}']}:** {row[f'detalhe_{i}']}" for i in range(1, 4)]
    return "\n".join(lines)


t, c, a = m["test"], m["cutoff"], m["cutoff"]["assumptions"]
naive, chosen = t["policies"]["cutoff_050"], t["policies"]["chosen"]
cut = num(t["cutoff"], 2)
top_reason, top_share = next(iter(t["top1_reason_share"].items()))

readme = f"""# Credit scoring com otimização de cutoff e explicabilidade

Modelo de concessão de crédito para uma cooperativa, do dado bruto à decisão: scorecard tradicional (regressão logística com WoE), LightGBM com restrições de negócio, **escolha do ponto de corte pelo lucro** e **motivos de recusa automáticos** para atender à LGPD.

> **Na base de teste, trocar o cutoff padrão de 0,50 por {cut} reduz as perdas com inadimplência em {brl(t["losses_avoided_vs_050"])} ({pct(t["losses_avoided_pct"])}), abrindo mão de só {pct(t["margin_lost_pct"], 1)} da margem.** O lucro da carteira sobe {pct(t["profit_gain_pct"], 1)}, ou {brl(t["profit_gain_vs_050_per_10k"])} a cada 10 mil propostas.

![Perdas e margem no teste](reports/figures/test_losses_vs_margin.png)

## O problema

Uma cooperativa quer automatizar a concessão de crédito. Aprovar demais gera perdas com inadimplência, e recusar demais afasta bons pagadores. O modelo precisa ordenar bem o risco, a decisão precisa fazer sentido financeiro, e cada recusa precisa ser explicável ao cooperado (art. 20 da LGPD).

**Dados:** [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit), com {num(m["data"]["n_rows"])} tomadores e taxa de default de {pct(m["data"]["default_rate"], 1)} (atraso de 90+ dias nos 2 anos seguintes). Divisão estratificada em treino, validação e teste (70/15/15). O teste só foi usado uma vez, no fim.

## Resultados (base de teste)

| Modelo | AUC | Gini | KS |
|---|---|---|---|
| Scorecard (logística + WoE, 7 variáveis) | {num(t["logistic"]["auc"], 3)} | {num(t["logistic"]["gini"], 3)} | {num(t["logistic"]["ks"], 3)} |
| **LightGBM (15 variáveis, com restrições monotônicas)** | **{num(t["lightgbm"]["auc"], 3)}** | **{num(t["lightgbm"]["gini"], 3)}** | **{num(t["lightgbm"]["ks"], 3)}** |

| Política (LightGBM) | Aprovação | Inadimplência da carteira | Perdas | Lucro |
|---|---|---|---|---|
| Aprovar todos | {pct(t["policies"]["approve_all"]["aprovação"], 1)} | {pct(t["policies"]["approve_all"]["inadimplência_carteira"], 2)} | {brl_mi(t["policies"]["approve_all"]["perdas"])} | {brl_mi(t["policies"]["approve_all"]["lucro"])} |
| Cutoff 0,50 (padrão) | {pct(naive["aprovação"], 1)} | {pct(naive["inadimplência_carteira"], 2)} | {brl_mi(naive["perdas"])} | {brl_mi(naive["lucro"])} |
| **Cutoff {cut} (escolhido na validação)** | **{pct(chosen["aprovação"], 1)}** | **{pct(chosen["inadimplência_carteira"], 2)}** | **{brl_mi(chosen["perdas"])}** | **{brl_mi(chosen["lucro"])}** |

Os 10% de propostas com menor score concentram {pct(t["bad_share_lowest_decile"])} dos inadimplentes.

## O que eu destacaria

**1. O cutoff certo vale mais do que o algoritmo.** Com as premissas usadas (ticket de {brl(a["ticket"])}, margem de {pct(a["margin_rate"])} para bons pagadores, EAD de {pct(a["ead_rate"])} e LGD de {pct(a["lgd"])}), aprovar só compensa quando a probabilidade de default é menor que ganho ÷ (ganho + perda) = {pct(a["breakeven_pd"], 1)}. O cutoff ótimo encontrado empiricamente na validação ({num(c["optimal_cutoff"], 2)}) caiu em cima desse valor teórico, o que indica que as probabilidades do modelo estão bem calibradas. Na validação, o LightGBM rendeu {brl(c["policies"]["optimal_lightgbm"]["lucro"] - c["policies"]["optimal_scorecard"]["lucro"])} a mais que o scorecard, e ajustar o cutoff rendeu {brl(c["profit_gain_vs_050"])}.

![Curva de lucro por cutoff](reports/figures/cutoff_profit_curve.png)

**2. Olhar os dados antes do modelo evitou erros sérios.** No EDA encontrei:
- valores 96 e 98 nas colunas de atraso que eram códigos de sistema, não contagens (com eles, as três colunas pareciam quase perfeitamente correlacionadas entre si);
- um `debt_ratio` que vira o valor absoluto da dívida quando a renda não é informada;
- renda faltante que não é aleatória.

Cada achado virou uma regra testada em `src/credit_scoring/features.py`.

**3. Restrições de negócio sem custo de desempenho.** O LightGBM foi obrigado a respeitar direções como "mais atraso nunca reduz o risco" e "mais idade nunca aumenta o risco". O AUC na validação cruzada ficou praticamente igual ao do modelo livre, e as explicações ficam coerentes com o que um analista de crédito esperaria.

**4. Motivos de recusa automáticos, em português.** O SHAP decompõe cada decisão, as variáveis são agrupadas em temas e os três que mais tiraram pontos viram motivos com o dado do próprio cliente. Na validação, o motivo principal do LightGBM está entre os três do scorecard em {pct(m["explainability"]["lgbm_top1_in_scorecard_top3"], 1)} dos casos. No teste, "{top_reason}" é o motivo principal em {pct(top_share)} das recusas. Exemplo real de uma recusa da validação:

{example_letter()}

![Efeito de cada variável (SHAP)](reports/figures/shap_beeswarm.png)

**5. Um ponto levantado para compliance.** A idade aparece entre os motivos em {pct(m["explainability"]["age_in_reasons_share"])} das recusas. Ela reflete um risco real nos dados, mas toca no princípio da não discriminação da LGPD. O notebook 06 mede o efeito e deixa a decisão para a área responsável, em vez de escondê-la.

## Limitações

- **As premissas financeiras são ilustrativas**, porque a base não tem valor, taxa nem recuperação. Elas ficam em `ProfitAssumptions` e a análise de sensibilidade mostra que a conclusão se mantém em vários cenários.
- **Não há dimensão de tempo na base**, então a validação é aleatória, não *out-of-time*. Em produção eu validaria em safras posteriores e monitoraria a estabilidade (PSI).
- **Só existem dados de quem recebeu crédito.** Não há como corrigir o viés de seleção (*reject inference*) com esta base.
- Os dados são de tomadores dos EUA de uma competição de 2011. Os números servem para demonstrar o método, não como referência de mercado brasileiro.

## Estrutura

```
notebooks/
  01_eda.ipynb                  análise exploratória e problemas de qualidade
  02_features.ipynb             limpeza, novas variáveis, divisão e Information Value
  03_logistic_scorecard.ipynb   coarse classing, seleção de variáveis e scorecard de pontos
  04_lightgbm.ipynb             LightGBM monotônico, comparação e calibração
  05_cutoff.ipynb               simulação financeira e escolha do ponto de corte
  06_explainability.ipynb       SHAP, motivos de recusa e checagens
  07_final_test.ipynb           avaliação única na base de teste
src/credit_scoring/             código reutilizado pelos notebooks (features, WoE, scorecard, LightGBM, cutoff, explicações)
tests/                          testes unitários (pytest)
reports/                        figuras, scorecard, motivos de recusa e metrics.json (fonte dos números deste README)
scripts/                        run_all.sh (reproduz tudo) e build_readme.py (gera este README)
```

## Como reproduzir

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12.

1. Baixe `cs-training.csv` e `cs-test.csv` da [competição no Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) e coloque em `data/raw/`. Com a CLI do Kaggle configurada:
   ```bash
   kaggle competitions download -c GiveMeSomeCredit -p data/raw && unzip data/raw/GiveMeSomeCredit.zip -d data/raw
   ```
2. Rode tudo (instala dependências, roda os testes, executa os notebooks em ordem e regenera este README):
   ```bash
   bash scripts/run_all.sh
   ```

**Stack:** pandas, scikit-learn, LightGBM, SHAP, matplotlib/seaborn, pytest.
"""

(ROOT / "README.md").write_text(readme)
print("README.md gerado")
