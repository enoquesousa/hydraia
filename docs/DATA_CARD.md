# Data Card — séries hidrológicas HydraIA

## Resumo

O dataset de treinamento é derivado de séries diárias públicas expostas pela API de mananciais da SABESP. Os dados brutos não são redistribuídos neste repositório; devem ser obtidos novamente na fonte e submetidos ao pipeline de preparação.

## Escopo usado pelo modelo

| Dimensão | Valor |
|---|---|
| Sistemas-alvo | Cantareira, Alto Tietê, Guarapiranga, Cotia, Rio Grande e Rio Claro |
| Período do artefato auditado | 01/01/2000 a 10/04/2026 |
| Frequência | diária |
| Variáveis observadas | chuva, volume útil, vazão natural, afluente, captada e jusante |
| Variáveis temporais derivadas | seno/cosseno do dia do ano e mês normalizado |
| Features finais | 39 |
| Targets | volume útil percentual de 6 sistemas |
| Lookback / horizonte | 90 / 30 dias |

São Lourenço é coletado no estado atual, mas não integra o modelo supervisionado porque sua série começa em 2018 e não cobre todo o recorte de treinamento.

## Proveniência e aquisição

1. `scripts/baixar_base_sabesp.py` consulta os endpoints históricos.
2. Os JSONs são armazenados localmente em `storage/raw/sabesp/series/sistemas/`.
3. `src.data.preparar_dataset` seleciona as 36 séries, alinha datas e constrói as partições.
4. O repositório mantém somente schema, código, metadados e hashes; não atribui licença própria aos dados da SABESP.

## Qualidade observada

- As seis séries de volume útil têm cobertura diária no artefato local.
- Algumas covariáveis começam depois de 2000; por exemplo, a chuva do Alto Tietê disponível localmente começa em dezembro de 2004.
- Há valores de volume útil abaixo de 0% (uso de volume morto) e acima de 100% em séries oficiais; não devem ser descartados sem interpretação hidrológica.
- Vazões podem conter valores negativos. Sua validade física e semântica deve ser confirmada junto ao dicionário da fonte.
- A preparação atual interpola lacunas de até sete dias e depois aplica `ffill/bfill` dentro da partição.

## Partições

| Partição | Intervalo calendário | Uso |
|---|---|---|
| Treino | até 31/12/2022 | ajuste de scalers e pesos |
| Validação | ano de 2023 | early stopping e resíduos |
| Teste | 01/01/2024 em diante | avaliação final |

Como cada partição cria suas próprias janelas, os primeiros 90 dias e os últimos 29 dias de cada intervalo não viram origens válidas.

## Usos adequados

- pesquisa de previsão multivariada de volume em protótipo;
- comparação de estratégias de integração entre ML e LLMs;
- ensino de pipelines temporais e proveniência.

## Usos inadequados

- decisão operacional sem validação por hidrólogos e dados oficiais atualizados;
- inferência causal sobre clima, gestão ou desmatamento;
- generalização para bacias fora da Macrometrópole;
- redistribuição dos dados sem verificar termos da fonte.

## Riscos e melhorias

Uma reprodução mais rigorosa deve versionar um manifesto de aquisição, validar unidades e limites físicos, usar imputação causal, registrar as datas de origem das janelas, manter hashes de cada série e avaliar mudança de regime.
