# Validação operacional real — 2026-08-08

Este registro documenta um ensaio pós-entrega feito para validar o perfil de
publicação do repositório. Ele não substitui os experimentos do TCC nem deve ser
apresentado como evidência histórica produzida na época da entrega.

## Ambiente

- Windows, Python 3.12 e execução reproduzível com `uv`;
- NVIDIA GeForce RTX 3060 com 12 GB de VRAM;
- LM Studio com Qwen3.5-35B-A3B Q4_K_M, contexto de 16.384 tokens,
  paralelismo 1 e identificador estável `hydraia-synth`;
- 40% das camadas do modelo local descarregadas na GPU;
- aproximadamente 11,4 GB de VRAM ocupados com o modelo carregado.

Durante as inferências locais, a utilização observada da GPU ficou entre 59% e
76%, o consumo entre 78 W e 86,6 W e a maior temperatura observada foi 51 °C.
Esses valores são observações desse equipamento, não requisitos mínimos nem um
benchmark controlado.

## Protocolo

O preflight estrito confirmou OpenWeather, API SABESP, artefatos HydroLSTM,
LM Studio, Gemini e OpenAI antes da execução. O teste de aceitação foi iniciado
com:

```powershell
uv --system-certs run --python 3.12 --no-project `
  --with-requirements requirements.txt `
  python -m src.cli run --economico --sem-baseline-fallback `
  --skip-preflight --horizonte-dias 7 `
  --cid real-20260808-publicacao
```

O `--skip-preflight` evita repetir as chamadas já aprovadas imediatamente antes
do teste; não relaxa o contrato da execução. `--sem-baseline-fallback` exige o
checkpoint HydroLSTM e faz a execução falhar em vez de substituir silenciosamente
o modelo por uma linha de base.

## Resultado observado

- término com código de saída zero em 93,2 s de relógio e 91,78 s medidos pela
  aplicação;
- fonte SABESP marcada como real, com data de referência 2026-08-08;
- seis sistemas e sete dias por sistema, totalizando 42 linhas de previsão;
- checkpoint HydroLSTM com SHA-256 abreviado `5a690ff80370`, sem fallback;
- especialistas efetivos: Gemini 2.5 Pro e GPT-5.6 Terra;
- sintetizador local: Qwen3.5-35B-A3B;
- ensemble classificado como convergente, com risco 3/5;
- volume médio simples observado dos seis sistemas: 60,55%;
- nenhum sistema abaixo de 30% na coleta real usada.

O escore de confiança do ensemble é uma medida heurística de concordância entre
agentes, e as probabilidades de evento são frequências empíricas do procedimento
de Monte Carlo. Nenhuma delas deve ser interpretada como probabilidade calibrada.
O resultado é um artefato de pesquisa e não deve ser a única base de uma decisão
operacional.

## Evidências locais

As saídas abaixo permanecem em `storage/`, que é ignorado pelo Git por poder
conter dados de terceiros, caches e informações operacionais:

- `relatorio_20260808_205824.md` — SHA-256
  `acd13a9a78c8933554d680a51c1a9b566fa229da7df9cd5db418df6b02d5a0e4`;
- `metadata_20260808_205824.json` — SHA-256
  `4592e16144186f8a93cbc26bed934b7ed02710c1421a70868d786b539f65f710`.

O relatório passou por uma auditoria automatizada das oito seções, contagem das
42 linhas, proveniência real, modelos efetivos e ausência de padrões de segredo.
A revisão do código foi encerrada com 297 testes aprovados e Ruff sem violações.

## Correções reveladas pelo teste

O ensaio de ponta a ponta também validou correções que testes isolados não
capturavam integralmente:

- confiança TLS integrada ao armazenamento de certificados do Windows, sem
  desativar a verificação HTTPS;
- OpenAI migrada para a Responses API e registro do modelo efetivamente usado;
- orçamento de saída e JSON estruturado ajustados para evitar respostas vazias;
- execução eager do HydroLSTM como padrão no Windows, sem depender de `cl.exe`;
- rejeição de síntese local truncada por limite de tokens;
- seções factuais renderizadas deterministicamente e narrativa restrita às
  seções analíticas;
- normalização Unicode de `Alto Tietê`, garantindo suas sete linhas no contrato
  de previsão.

Para interpretar as diferenças em relação ao trabalho entregue, consulte
[Errata científica](ERRATA.md). Para reproduzir o perfil local, consulte
[Modelo local e GPU](LOCAL_MODEL.md).
