# Model Card — HydroLSTM

> **International academic context:** this model card is part of HydraIA, a
> [Brazilian Undergraduate Thesis (TCC)](ACADEMIC_CONTEXT.md) and final-year
> Computer Engineering capstone project completed at FMU, Brazil, in 2026.

## Identificação

- **Nome:** HydroLSTM
- **Tipo:** LSTM bidirecional multivariada, many-to-many por cabeça direta
- **Framework do artefato:** PyTorch 2.7.1
- **Entrada:** `(batch, 90, 39)`
- **Saída:** `(batch, 30, 6)`
- **Parâmetros:** 2.277.556

## Arquitetura

Duas camadas bidirecionais, 256 unidades ocultas por direção, dropout de 0,25 entre camadas e na representação final, seguidas de camada linear `512 → 180`. A bidirecionalidade percorre apenas os 90 dias históricos da janela; não acessa os 30 targets futuros diretamente.

## Objetivo e treinamento

O modelo prevê conjuntamente o volume útil percentual de seis sistemas. Foi otimizado com AdamW e função `0,6·MAE + 0,4·MSE` em targets padronizados, seed 42, clipping de gradiente e early stopping.

## Avaliação reproduzida

| Partição | N | MAE (pp) | RMSE (pp) |
|---|---:|---:|---:|
| Validação | 246 | 7,0377 | 8,7970 |
| Teste | 712 | 6,1928 | 7,5718 |

Ambiente: Python 3.12, `torch 2.7.1+cpu`, `numpy 2.2.6`, `scikit-learn 1.8.0`. Os hashes estão em `artifacts/model_evaluation.json`.

O `hydro_lstm_meta.json` local era anterior ao checkpoint e não foi publicado como evidência canônica. A diferença entre as métricas históricas e as reproduzidas é documentada em `docs/ERRATA.md`.

## Baselines

O código implementa persistência, drift linear e climatologia mensal. O artefato histórico não preserva datas de origem por amostra no NPZ, de modo que a comparação sazonal completa não foi reproduzida. Não há evidência versionada de superioridade sobre ARIMA/SARIMA, modelos físicos ou Transformers.

## Incerteza

Intervalos p10/p50/p90 e probabilidades de limiares usam resíduos empíricos do split de validação. Eles não têm garantia de cobertura e podem perder validade sob mudança de regime. O cache de resíduos deve ser invalidado quando dataset, scaler ou checkpoint mudarem.

## Uso pretendido

Pesquisa, demonstração acadêmica e geração de sinais auxiliares para interpretação humana. O modelo não é aprovado para operação de abastecimento, alertas públicos ou alocação de recursos.

## Limitações conhecidas

- seis sistemas, sem São Lourenço no target;
- recorte geográfico único;
- variáveis exógenas climáticas ausentes do modelo;
- imputação não estritamente causal;
- trajetórias curtas achatadas;
- artefatos históricos sem manifesto de linhagem único;
- saída limitada por clipping de inferência a `[0,150]`, enquanto a semântica operacional de valores negativos merece tratamento específico.

## Recomendações antes de novo release

1. Corrigir imputação causal e regenerar dataset/pesos.
2. Persistir manifesto com hashes, dependências, seed e commit.
3. Comparar baselines com o mesmo protocolo e intervalos de confiança.
4. Medir erro por sistema, horizonte, estação e regime hidrológico.
5. Calibrar intervalos e probabilidades em conjunto independente.
