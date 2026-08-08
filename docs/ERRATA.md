# Errata científica e notas pós-entrega

O PDF do TCC é preservado como entregue. Este documento registra descobertas da auditoria de publicação feita em 08/08/2026 e prevalece para interpretação dos artefatos atuais.

## 1. Unidade do MAE de validação

O artigo apresenta `MAE de validação = 0,35 pp`. No código de treinamento, esse valor era calculado antes do `inverse_transform`, no espaço padronizado do `StandardScaler`. Portanto, `0,35` é adimensional e não pode ser comparado diretamente ao MAE de teste em pontos percentuais.

Ao reavaliar o checkpoint e o dataset presentes no workspace com PyTorch 2.7.1, NumPy 2.2.6 e scikit-learn 1.8.0, obtiveram-se 7,04 pp de MAE e 8,80 pp de RMSE na validação.

## 2. Linhagem dos artefatos

O dataset, os scalers e o checkpoint têm data de 13/05/2026, enquanto `hydro_lstm_meta.json` tem data de 20/04/2026. O arquivo de métricas não descreve com segurança o checkpoint mais recente.

Para o par atual, a auditoria obteve 6,19 pp de MAE e 7,57 pp de RMSE no teste, em vez de 6,05/7,53 registrados no metadado antigo. Veja `artifacts/model_evaluation.json` para hashes e ambiente.

## 3. “Confiança” do ensemble

O percentual produzido pelo ensemble é derivado de autoconfianças fornecidas pelos próprios LLMs e regras manuais de bônus/penalização. Não há calibração contra frequências observadas, avaliação humana ou desfechos reais. O termo tecnicamente adequado é **escore heurístico de concordância**, não probabilidade ou grau de confiança estatístico.

## 4. Redução de tokens e custo

As afirmações de redução superior a 90% e de economia equivalente não são acompanhadas por benchmark versionado com contagem de tokens, prompts de referência, preços e repetição experimental. Devem ser tratadas como observações de desenvolvimento até que um protocolo reproduzível seja publicado.

## 5. Quantidade de testes

O artigo registra 265 casos e um subconjunto de 81; o README anterior registrava 58. A árvore auditada coletou 283 casos antes das correções de publicação e 289 após a integração local preparada. Quantidade de testes é evidência de cobertura de regras, não substituto para cobertura de código nem validação científica externa.

## 6. Fontes realmente ativas

O objetivo inicial cita SABESP, OpenWeather, DAEE e ANA. A implementação final ativa consome SABESP e OpenWeather; DAEE e ANA aparecem no contexto bibliográfico, não no caminho principal de execução.

## 7. Diagrama entregue

Na figura do artigo, a seta do OpenWeather pode sugerir entrada na HydroLSTM. Na implementação, a HydroLSTM usa apenas 36 séries da SABESP e três variáveis temporais. OpenWeather alimenta o especialista meteorológico.

## 8. Imputação temporal

Os scalers são ajustados apenas no treino e as partições não se contaminam entre si. Entretanto, `interpolate(method="time")` e `bfill` usam observações posteriores dentro de uma mesma partição para preencher lacunas anteriores. Uma avaliação estritamente causal deve usar somente informação disponível até cada origem de previsão e retrain/reavaliar os artefatos.

## 9. Intervalos e probabilidades

Quantis e probabilidades de limiar são calculados ao somar resíduos empíricos de validação à previsão. Não há teste de cobertura, calibração, independência ou mudança de regime. As saídas devem ser chamadas estimativas empíricas exploratórias.

## 10. Validade externa

Não houve comparação completa e versionada com ARIMA/SARIMA, modelos físico-hidrológicos ou Transformers, nem painel sistemático de especialistas para os relatórios. O trabalho demonstra viabilidade de protótipo e engenharia de integração; superioridade científica sobre alternativas permanece hipótese aberta.

## 11. Modelo local após a entrega

O artigo entregue registra Mistral 7B Instruct v0.3 como sintetizador. Para a publicação do repositório, o perfil operacional local foi atualizado para Qwen3.5-35B-A3B Q4_K_M, mantendo a mesma API OpenAI-compatible do LM Studio. Essa troca é uma melhoria de implantação pós-entrega e não foi usada para produzir as análises ou métricas históricas do TCC; uma comparação controlada entre sintetizadores ainda não foi realizada.
