# Contribuindo

> HydraIA is a completed
> [Brazilian Undergraduate Thesis (TCC)](docs/ACADEMIC_CONTEXT.md) and final-year
> Computer Engineering capstone project. Contributions must preserve its
> academic provenance and the boundary between submitted and post-submission work.

Este repositório registra um trabalho acadêmico concluído. Correções de reprodutibilidade, segurança, documentação e testes são bem-vindas; mudanças que alterem resultados científicos devem vir acompanhadas de nova avaliação e proveniência dos artefatos.

## Fluxo local

1. Crie um ambiente Python 3.12.
2. Instale `requirements.txt` e `requirements-dev.txt`.
3. Execute `python -m pytest -q`.
4. Execute `ruff check src tests scripts`.
5. Não inclua credenciais, dados brutos, pesos, caches ou documentos administrativos.

## Critérios para resultados

Relate, no mínimo: hash do dataset e do checkpoint, versões do Python/PyTorch/NumPy/scikit-learn, seed, partição temporal, número de janelas e métricas em unidades físicas. Diferencie resultados reproduzidos, resultados históricos do artigo e hipóteses ainda não validadas.

Ao contribuir, você não recebe automaticamente licença de reutilização sobre o restante do projeto. Consulte `LICENSE`.
