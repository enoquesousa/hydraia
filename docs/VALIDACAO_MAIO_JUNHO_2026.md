# Auditoria da previsão de maio para junho de 2026

## Conclusão

A afirmação de que a previsão foi certeira é sustentada em um escopo específico:
o produto emitido em 7 de maio apresentou excelente aderência para o Cantareira
nos sete dias seguintes e antecipou corretamente a direção predominante dos
volumes até o começo de junho. Isso é evidência operacional relevante, mas não
autoriza afirmar acerto universal ou probabilidades calibradas.

Esta é uma auditoria retrospectiva de um produto de previsão arquivado. Os
relatórios têm timestamps internos, datas de modificação locais e hashes atuais,
mas não foram encontrados um carimbo de tempo independente ou um commit público
anterior ao período observado. Portanto, o resultado não é descrito como ensaio
prospectivo pré-registrado.

## Fontes e protocolo

- produto arquivado: `relatorio_20260507_233250.md`, emitido com referência em
  7 de maio de 2026;
- metadados arquivados: `metadata_20260507_233250.json`;
- observações posteriores: endpoint oficial de resumo diário do
  [Portal dos Mananciais da SABESP](https://mananciais.sabesp.com.br/base-dados);
- data da auditoria: 8 de agosto de 2026, horário de Brasília;
- unidade dos erros: pontos percentuais de volume útil.

Os arquivos operacionais continuam fora do Git, conforme o escopo de publicação.
Seus hashes, os valores observados e as métricas estão preservados no artefato
[`forecast_audit_may_june_2026.json`](../artifacts/forecast_audit_may_june_2026.json).

## Curto prazo: Cantareira, 7 a 14 de maio

O relatório de 7 de maio publicou p50 de 41,9% para os sete dias. A comparação
com os valores posteriormente retornados pela SABESP foi:

| Data observada | Publicado (%) | Observado (%) | Erro absoluto (pp) |
|---|---:|---:|---:|
| 08/05 | 41,90 | 41,7246 | 0,1754 |
| 09/05 | 41,90 | 41,5618 | 0,3382 |
| 10/05 | 41,90 | 41,5007 | 0,3993 |
| 11/05 | 41,90 | 41,5612 | 0,3388 |
| 12/05 | 41,90 | 41,5371 | 0,3629 |
| 13/05 | 41,90 | 41,4194 | 0,4806 |
| 14/05 | 41,90 | 41,3217 | 0,5783 |

**MAE: 0,3819 pp; RMSE: 0,3994 pp; erro máximo: 0,5783 pp.** Para esse
sistema, emissão e horizonte, o desempenho foi excelente.

## Direção até 6 de junho

O relatório recomendava preparação para uma possível redução gradual dos
volumes. Entre 7 de maio e 6 de junho, cinco dos seis sistemas modelados caíram:

| Sistema | 07/05 (%) | 06/06 (%) | Delta (pp) | Compatível com queda |
|---|---:|---:|---:|:---:|
| Cantareira | 41,8524 | 39,6414 | -2,2110 | Sim |
| Alto Tietê | 52,4784 | 50,2324 | -2,2460 | Sim |
| Guarapiranga | 81,2889 | 76,5494 | -4,7395 | Sim |
| Cotia | 62,1521 | 60,5461 | -1,6061 | Sim |
| Rio Grande | 88,7763 | 85,3058 | -3,4705 | Sim |
| Rio Claro | 61,4054 | 64,8794 | +3,4741 | Não |

A compatibilidade direcional foi de **5/6 (83,3%)**. Esse número descreve a
direção de um único episódio; não é uma estimativa populacional da acurácia do
modelo.

## O que não pode ser chamado de acerto preciso

O horizonte h+30 mostrado no relatório era uma distribuição climatológica
empírica, não a previsão pontual do HydroLSTM. O valor observado em 6 de junho
ficou entre p10 e p90 nos seis sistemas, porém:

- a largura média dos intervalos era 46,32 pp;
- o MAE das medianas p50 foi 15,11 pp.

Logo, cobertura de 100% com esses intervalos não prova boa calibração ou precisão.

Também foi auditado o relatório de 13 de maio. Sua tabela publicou p50 de 53,5%
para o Cantareira, enquanto os valores de 14 a 20 de maio ficaram entre 41,06% e
41,32%, resultando em MAE de 12,34 pp. Os metadados antigos registram
`probabilidades_previsao: null`, e naquela versão as tabelas eram redigidas pela
LLM sem persistência do vetor numérico bruto. Assim, não é possível distinguir
retrospectivamente erro do modelo de erro de síntese. A renderização determinística
introduzida na versão de publicação elimina essa ambiguidade para execuções futuras.

## Interpretação científica

O achado defensável é: **houve um acerto numérico forte no teste de curto prazo
de 7 de maio e um acerto direcional relevante até junho**. O mesmo conjunto de
evidências também mostra por que uma única execução não basta para declarar o
sistema “certeiro”. Uma avaliação prospectiva mais forte deve congelar antes do
evento o vetor bruto, hashes, checkpoint, código e timestamp externo, e depois
calcular MAE, RMSE, cobertura, largura dos intervalos e calibração em várias
janelas consecutivas.
