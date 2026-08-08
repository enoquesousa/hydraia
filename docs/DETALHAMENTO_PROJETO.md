# Detalhamento do Projeto HydraIA

Este documento detalha a identidade, propósito, arquitetura e tecnologias do projeto HydraIA, servindo como base descritiva para apresentações e capítulos introdutórios do TCC.

---

## 1. Identidade do Projeto

### Nome
**HydraIA** (Hydra + IA)
- *Hydra*: Referência à água (hidro) e à resiliência (mitologia).
- *IA*: Inteligência Artificial, o motor central do sistema.

### Conceito
Um sistema de análise de risco hídrico que integra dados em tempo real da SABESP, previsão de volume via deep learning e interpretação técnica por ensemble de três grandes modelos de linguagem. O objetivo não é apenas informar "se vai chover", mas cruzar a chuva prevista com o histórico de 26 anos dos reservatórios e estimar o impacto real no abastecimento da Macrometrópole de São Paulo.

### Visão
Democratizar o acesso à análise de risco hídrico de alta precisão, produzindo relatórios técnicos acionáveis para gestores da SABESP e pesquisadores a partir de dados públicos e modelos de IA rodando em hardware acessível.

---

## 2. Problema e Objetivos

### O Problema
Dados de reservatórios estão dispersos em portais governamentais sem contexto histórico. Previsões meteorológicas fornecem mm de chuva sem traduzir o impacto no abastecimento. A combinação de séries temporais longas com análise qualitativa de risco requer especialização que sistemas automáticos não ofereciam de forma integrada.

### Objetivo Principal
Desenvolver um pipeline que, a partir dos dados reais da API SABESP, produza um relatório técnico de risco hídrico com escore explícito de concordância — usando aprendizado de máquina para a previsão de volume e ensemble de LLMs para a interpretação.

### Objetivos Específicos
1. **Coletar dados reais**: Consumir a API oficial SABESP em tempo real (não estimativas).
2. **Prever com ML**: Usar um LSTM treinado em 26 anos de histórico para projetar o volume dos reservatórios nos próximos 7 dias.
3. **Cruzar com meteorologia**: Integrar previsão do OpenWeather para avaliar o impacto da chuva esperada nos reservatórios.
4. **Validação cruzada entre LLMs**: Dois especialistas independentes (Gemini e GPT) analisam perspectivas diferentes do mesmo cenário; divergências são sinalizadas ao modelo de síntese.
5. **Síntese local e privada**: O Qwen3.5-35B-A3B (com offload na GPU local) produz o relatório final sem expor o corpus de contexto à nuvem.

---

## 3. Arquitetura do Pipeline

O pipeline executa em quatro camadas sequenciais:

### Camada 1 — Coleta de Dados
- **`src/api/sabesp.py`**: Consulta `/api/v4/dados/ultima-data` para obter a data mais recente publicada pela SABESP, depois `/api/v4/sistemas/dados/resumo-diario/{data}` para coletar volume, chuva e vazões dos 7 sistemas da Macrometrópole. Retry automático (3 tentativas) com backoff de 1s.
- **`src/api/openweather.py`**: Coleta 8 dias de previsão diária (temperatura, umidade, precipitação) via One Call 3.0. Não bloqueia o pipeline se falhar.
- **Cache local**: SABESP e OpenWeather têm cache de 1 hora em `storage/raw/cache/` para evitar chamadas repetidas à API durante desenvolvimento.

### Camada 2 — Previsão ML
- **`src/data/inferencia_ts.py`**: Carrega o modelo `hydro_lstm_melhor.pt` e os últimos 90 dias de dados reais. Gera previsão de volume útil (%) para 6 sistemas nos próximos 30 dias (pipeline usa os 7 primeiros).
- **HydroLSTM**: LSTM bicamada (hidden=256, dropout=0.25) treinada em séries diárias da SABESP. No par checkpoint/dataset auditado: MAE de validação 7,04 pp e MAE de teste 6,19 pp; consulte `docs/ERRATA.md`.

### Camada 3 — Ensemble LLM
Implementada em `src/core/orquestrador.py`:

**Fase 1 — Análises paralelas** (ThreadPoolExecutor, 2 workers):
- Gemini 2.5 Pro recebe dados SABESP + projeção HydroLSTM → analisa risco hidrológico → JSON com `nivel_risco` (1-5), `tendencia`, `analise`, `confianca_pct`.
- GPT recebe dados SABESP + HydroLSTM + OpenWeather → analisa impacto meteorológico → JSON com `nivel_risco`, `chuva_total_esperada_mm`, `impacto_nos_reservatorios`, `confianca_pct`.

**Fase 2 — Cross-validation**:
- Delta ≤ 1 ponto → CONVERGÊNCIA → confiança bonificada em 15%.
- Delta > 1 ponto → DIVERGÊNCIA → confiança penalizada em 10% por ponto de delta. O sintetizador local recebe a divergência explicitamente.
- Apenas 1 modelo → PARCIAL → confiança penalizada em 30%.
- Risco consolidado: sempre o maior entre os dois (princípio conservador).

**Fase 3 — Síntese local**:
- Qwen3.5-35B-A3B Q4_K_M (LM Studio, porta 1234, CUDA + RAM) sintetiza as análises dos dois especialistas em relatório técnico estruturado em PT-BR.

### Camada 4 — Saída
- Relatório impresso no console via logging.
- Relatório salvo em `storage/outputs/reports/relatorio_TIMESTAMP.md`.
- Metadados (risco, confiança, data de referência) salvos em `metadata_TIMESTAMP.json`.

---

## 4. Modelos de IA

### HydroLSTM (local)
- **Tipo**: LSTM bicamada (PyTorch)
- **Função**: Previsão de volume útil (v_util em %) dos reservatórios
- **Dados de treino**: 26 anos de séries diárias da SABESP (2000–2025)
- **Features**: 39 (volume, chuva, vazão de todos os sistemas)
- **Targets**: 6 sistemas (Cantareira, Alto Tietê, Guarapiranga, Cotia, Rio Grande, Rio Claro)
- **Lookback**: 90 dias | **Horizonte**: 30 dias
- **Execução**: GPU RTX 3060 (CUDA 11.8)

### Qwen3.5-35B-A3B Q4_K_M (local)
- **Status**: perfil de implantação pós-entrega; o artigo original usou Mistral 7B Instruct v0.3
- **Função**: Síntese final do relatório técnico
- **Arquitetura**: Mixture-of-Experts, 35B parâmetros totais e 3B ativos por token
- **Execução**: RTX 3060 via runtime CUDA do LM Studio, com camadas remanescentes em RAM
- **Vantagem**: Privacidade — dados brutos não saem da máquina local; custo zero por token

### Gemini 2.5 Pro (nuvem)
- **Função**: Especialista hidrológico (Fase 1 do ensemble)
- **Entrada**: Estado atual dos reservatórios + projeção HydroLSTM
- **Saída**: JSON com nível de risco, tendência e análise técnica

### GPT (nuvem)
- **Função**: Especialista meteorológico (Fase 1 do ensemble)
- **Entrada**: Estado dos reservatórios + HydroLSTM + previsão OpenWeather
- **Saída**: JSON com nível de risco, impacto nos reservatórios e chuva esperada
- **Quirk**: Modelos gpt-5 e o-series não aceitam parâmetro `temperature` — tratado com try/except

---

## 5. Estrutura de Arquivos Ativos

| Arquivo | Função |
|---|---|
| `src/api/sabesp.py` | Coletor SABESP — API v4 oficial |
| `src/api/openweather.py` | Coletor OpenWeather One Call 3.0 |
| `src/core/integracao.py` | Orquestrador do pipeline (coleta → ML → ensemble → saída) |
| `src/core/orquestrador.py` | Ensemble Gemini + GPT → validação determinística → síntese local |
| `src/data/inferencia_ts.py` | Inferência HydroLSTM |
| `src/data/preparar_dataset.py` | Prepara séries temporais a partir dos JSONs históricos |
| `src/data/treinar_modelo_ts.py` | Treina o HydroLSTM |
| `src/models/hydraia_llm.py` | Cliente model-agnostic para o servidor OpenAI-compatible do LM Studio |
| `src/utils/env_loader.py` | Carrega e valida variáveis do `.env` |
| `scripts/historico_analisador.py` | Análise histórica exploratória com o LLM local |
| `scripts/processar_historico_mistral.py` | Consolida histórico SABESP em base de conhecimento |

---

## 6. Tecnologias

### Linguagem
- Python 3.11+ — padrão de fato em ciência de dados e IA

### Machine Learning
- **PyTorch 2.7 + CUDA 11.8** — treino e inferência do HydroLSTM na GPU
- **NumPy / Pandas** — manipulação de séries temporais
- **scikit-learn** — normalização de features (StandardScaler)

### LLMs e APIs
- **google-genai 1.73** — SDK oficial Gemini (não `google-generativeai`)
- **openai 2.31** — SDK OpenAI
- **requests** — chamadas SABESP, OpenWeather e LM Studio

### Qualidade
- **pytest** — suíte offline cobrindo coleta, ensemble, cache, resiliência, CLI, baselines e pipeline simulado

---

## 7. Diferenciais

1. **Dados reais obrigatórios**: O pipeline recusa execução se a API SABESP não responder — não usa simulações nem dados fictícios.
2. **Validação cruzada explícita**: A divergência entre os LLMs especialistas é comunicada ao modelo de síntese e refletida em um escore heurístico de concordância do relatório.
3. **Síntese local e privada**: O relatório final é gerado pelo Qwen3.5 rodando localmente com aceleração CUDA — o corpus de síntese não sai da máquina.
4. **Contexto histórico profundo**: O HydroLSTM foi treinado em 26 anos de dados diários, permitindo capturar sazonalidades e padrões de longo prazo.
5. **Resiliência**: Cache de 1h para SABESP e OpenWeather; retry com backoff para os LLMs em nuvem; OpenWeather não bloqueia o pipeline se indisponível.
