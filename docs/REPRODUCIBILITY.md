# Guia de reprodutibilidade

## Níveis de reprodução

1. **Software:** executar testes e demo totalmente offline.
2. **Pipeline:** adquirir dados atuais, preparar dataset e treinar novo checkpoint.
3. **Artefato histórico:** avaliar exatamente o checkpoint/dataset auditados, caso o autor os forneça fora do Git.

## Ambiente recomendado

- Python 3.12;
- PyTorch 2.7.1;
- NumPy 2.2.6;
- Pandas 2.3.3;
- scikit-learn 1.8.0;
- seed `HYDRAIA_SEED=42`.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.cli run --modo-simulado --skip-llm --skip-preflight
```

## Obter e preparar dados

```bash
python scripts/baixar_base_sabesp.py
python -m src.data.preparar_dataset
```

Registre data/hora de aquisição, URLs, códigos HTTP e hashes dos JSONs. A API pode ser atualizada; reproduzir o código não garante recuperar o mesmo snapshot histórico.

## Treinar

```bash
python -m src.data.treinar_modelo_ts --epochs 150 --batch 128 --lr 0.001 --seed 42
```

O treinamento deve gerar `hydro_lstm_melhor.pt` e `hydro_lstm_meta.json` na mesma execução. Não copie metadados de outro checkpoint.

## Avaliar um artefato

Coloque localmente:

```text
storage/models/time_series/hydro_lstm_melhor.pt
storage/models/time_series/hydro_lstm_meta.json
storage/processed/datasets/sabesp_dataset.npz
storage/processed/datasets/scaler_X.pkl
storage/processed/datasets/scaler_y.pkl
```

Depois execute:

```bash
python scripts/avaliar_modelo.py
python scripts/avaliar_modelo.py --output artifacts/model_evaluation.local.json
```

## Hashes do par auditado em 08/08/2026

| Artefato | SHA-256 |
|---|---|
| dataset NPZ | `56e01c88fd1e7f4145cb3b72b02afd1309e5bba71b56a702a6f87a50f02ec771` |
| scaler y | `45fd0e42e6cdbf7d495759846554c5c890bbfd3a1561c0658097bee33f5fdb27` |
| checkpoint | `5a690ff80370454801c2e34dee721ab290589f8cac91b394125f9578688bd906` |
| metadado antigo | `b50dedf96950b49fb96f85afaca86192d9d6e22bd0d8620769dd390ae19fb187` |

## Pipeline completo com LLMs

Copie `.env.example` para `.env`, configure as chaves e inicie o LM Studio. Valide antes de gastar chamadas:

```bash
lms load qwen/qwen3.5-35b-a3b --gpu 0.40 --context-length 16384 --parallel 1 --identifier hydraia-synth --no-speculative-draft-mtp
lms server start --port 1234
```

```bash
python -m src.cli validate --strict --sondar-modelos
python -m src.cli run --economico
```

Registre os IDs exatos dos modelos, data da chamada, prompts, parâmetros, respostas e custos. Modelos hospedados mudam; o nome lógico sozinho não garante reprodução comportamental.

## Critérios de equivalência

- **Exata:** hashes de dados/checkpoint e versões iguais.
- **Numérica:** métricas dentro de tolerância declarada, com mesmo split.
- **Conceitual:** mesmo protocolo em nova coleta/novo treino, reportado como réplica e não como reprodução exata.
