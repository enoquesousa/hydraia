# Modelo local de síntese e uso de GPU

> **Mudança pós-entrega:** o artigo entregue registra Mistral 7B Instruct v0.3. O Qwen3.5 é o perfil operacional preparado posteriormente para a publicação do repositório; não deve ser apresentado como modelo usado nos resultados históricos do TCC.

## Perfil recomendado

Na máquina de referência — RTX 3060 12 GB, 48 GB de RAM e Intel Core i5-13400F — o sintetizador recomendado é:

| Item | Configuração |
|---|---|
| Modelo | Qwen3.5-35B-A3B |
| Quantização | GGUF Q4_K_M |
| Arquitetura | MoE, 35B parâmetros totais e 3B ativos por token |
| Contexto operacional | 16.384 tokens |
| Offload | 40% das camadas, ajustado empiricamente |
| Concorrência | 1 geração |
| Runtime | llama.cpp CUDA no LM Studio |
| Identificador da API | `hydraia-synth` |

A quantização ocupa aproximadamente 21,2 GB. Como não cabe integralmente nos 12 GB de VRAM, o LM Studio descarrega na GPU o máximo seguro e mantém as camadas restantes em RAM. Esse perfil privilegia qualidade de síntese e usa os dois níveis de memória; não implica que 100% dos pesos residam na GPU.

## Instalação

O executável `lms` acompanha o LM Studio. No Windows, se ele ainda não estiver no `PATH`, use `%USERPROFILE%\.lmstudio\bin\lms.exe`.

```bash
lms get qwen/qwen3.5-35b-a3b@q4_k_m --gguf --yes
lms load qwen/qwen3.5-35b-a3b --gpu 0.40 --context-length 16384 --parallel 1 --identifier hydraia-synth --no-speculative-draft-mtp --yes
lms server start --port 1234
```

No Windows, o mesmo perfil pode ser iniciado novamente com:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/iniciar_modelo_local.ps1
```

O estimador do LM Studio 1.0.6 apresentou confiança baixa para esse MoE e não diferenciou os percentuais de offload. Por isso, o valor foi validado por carga e inferência reais. Em outra máquina, comece abaixo desse percentual e monitore antes de aumentá-lo. Flash Attention deve permanecer habilitado no perfil quando suportado pelo runtime.

## Integração com o HydraIA

```dotenv
LM_STUDIO_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=hydraia-synth
LM_STUDIO_TIMEOUT=300
LM_STUDIO_REASONING=off
```

O cliente em `src/models/hydraia_llm.py` também aceita `LM_STUDIO_MODEL=auto` e ignora modelos de embedding durante a descoberta.

## Verificação

```bash
lms ps
python -m src.cli validate --strict --sondar-modelos
```

Além de confirmar que o processo usa o runtime CUDA, monitore VRAM, temperatura e potência com `nvidia-smi`. Utilização baixa entre requisições é normal; a carga deve subir durante a inferência.

## Medição de referência

Smoke test de 08/08/2026, com 94 tokens de entrada e 77 de saída:

| Medida | Resultado observado |
|---|---:|
| Velocidade de geração | 39,79 tokens/s |
| Tempo para primeiro token | 1,02 s |
| Utilização máxima da GPU | 100% |
| VRAM máxima do sistema | 11.955 / 12.288 MiB |
| Potência máxima | 78,2 W |
| Temperatura máxima | 45 °C |

Esses números são um teste operacional único, não um benchmark científico. Variam com prompt, processos gráficos, driver, versão do runtime e temperatura ambiente. O GGUF auditado tinha SHA-256 `f25d609171b8f80950a60f38696597f74025de4070682d2fb1eeffe306ca7d5d`.
