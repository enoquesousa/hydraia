# Artefatos pequenos de auditoria

Este diretório contém apenas metadados textuais adequados ao Git. Dados brutos, scalers, checkpoints e caches permanecem fora do repositório.

- `model_evaluation.json`: reprodução do checkpoint/dataset encontrados em 08/08/2026, com hashes e versões.
- `local_model_profile.json`: perfil pós-entrega do sintetizador local, hashes, hardware e smoke test CUDA.
- `forecast_audit_may_june_2026.json`: auditoria retrospectiva da previsão arquivada de maio contra observações SABESP posteriores.

Use `python scripts/avaliar_modelo.py --output artifacts/model_evaluation.local.json` para gerar uma avaliação de seus artefatos locais.
