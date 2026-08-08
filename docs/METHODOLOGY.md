# Metodologia e formulação matemática

## Problema supervisionado

Para cada instante de origem `t`, o modelo recebe uma janela retrospectiva de 90 dias:

```text
X_t ∈ R^(90 × 39)
```

As 39 variáveis correspondem a 36 séries observadas — chuva, volume útil e quatro vazões para seis sistemas — mais `sin(dia_do_ano)`, `cos(dia_do_ano)` e mês normalizado. A saída conjunta é:

```text
Ŷ_t ∈ R^(30 × 6)
```

Cada coluna prevê o volume útil percentual de um sistema nos 30 dias seguintes.

## Partição e normalização

- treino: até 31/12/2022;
- validação: 01/01/2023 a 31/12/2023;
- teste: a partir de 01/01/2024;
- `StandardScaler` ajustado apenas no treino;
- janelas criadas separadamente dentro de cada partição.

O código evita contaminação entre partições, mas a imputação temporal usa interpolação e `bfill` dentro de cada partição. Isso não é estritamente causal em relação a cada origem de previsão; é uma limitação metodológica registrada na errata.

## HydroLSTM

A rede possui duas camadas LSTM bidirecionais com 256 unidades por direção, dropout de 0,25 e cabeça linear de `512 → 180`. O total é 2.277.556 parâmetros treináveis.

A função objetivo no espaço padronizado é:

```text
L(ŷ, y) = 0,6 · MAE(ŷ, y) + 0,4 · MSE(ŷ, y)
```

O otimizador é AdamW, com `weight_decay=10⁻⁴`, clipping de gradiente em 1,0, scheduler cossenoidal e early stopping pelo MAE padronizado de validação.

MAE e RMSE publicáveis devem ser calculados após `inverse_transform`:

```text
MAE  = (1/n) Σ |y_i - ŷ_i|
RMSE = sqrt((1/n) Σ (y_i - ŷ_i)²)
```

## Baselines

- **Persistência:** `ŷ_(t+h) = y_t` para todo `h`.
- **Drift linear:** regressão linear nos 14 pontos mais recentes e extrapolação por horizonte.
- **Climatologia mensal:** média do mês no conjunto de treino; disponível no pipeline, mas o benchmark histórico salvo não contém o mês de origem por amostra e por isso não a compara de forma completa.

O procedimento chamado `backtest_rolling` no código calcula métricas por horizonte sobre janelas sobrepostas de teste com modelo fixo. Ele não retreina o modelo a cada origem e, portanto, é mais precisamente uma avaliação rolling-origin fixa.

## Incerteza por resíduos

Para sistema `j` e horizonte `h`, define-se o resíduo de validação:

```text
e_(i,h,j) = y_(i,h,j) - ŷ_(i,h,j)
```

Os intervalos empíricos são deslocamentos da previsão pontual:

```text
q_α(h,j | x) = ŷ_(h,j)(x) + Quantil_α({e_(i,h,j)})
```

E a probabilidade empírica de cruzar um limiar `τ` é:

```text
P(Y ≤ τ | x) ≈ (1/N) Σ 1[ŷ(x) + e_i ≤ τ]
```

Essas quantidades dependem de resíduos históricos exchangeáveis. Não são intervalos conformais e sua cobertura/calibração ainda não foi validada.

## Ensemble de LLMs

Cada especialista retorna nível de risco `r ∈ {1,…,5}` e autoconfiança declarada `c ∈ [0,100]`. Com duas respostas válidas:

```text
Δ = |r_hidro - r_meteo|
r_consolidado = max(r_hidro, r_meteo)
c_base = round((c_hidro + c_meteo) / 2)
```

O primeiro ajuste é:

```text
Δ ≤ 1: c₁ = min(100, round(1,15 · c_base))
Δ > 1: c₁ = max(0, c_base - 10 · Δ)
```

O escore semântico `s ∈ [0,1]` pondera diferença de risco (50%), coerência tendência/impacto (30%) e diferença de autoconfiança (20%). O escore final é:

```text
c_ensemble = clip(c₁ + round(25 · (s - 0,6)), 0, 100)
```

Com apenas um especialista, aplica-se `round(0,7 · c)`. Esses números são regras de engenharia, não estimativas probabilísticas calibradas. O nome recomendado para publicação é **escore de concordância do ensemble**.

## Validação de software

Os testes são offline, usam mocks para provedores e cobrem regras determinísticas. Eles demonstram conformidade da implementação com as regras codificadas; não demonstram validade hidrológica externa nem qualidade clínica/operacional do texto gerado.
