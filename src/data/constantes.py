"""Constantes compartilhadas pelo pipeline de séries temporais.

Definir aqui (Single Source of Truth) evita drift entre `preparar_dataset.py`
e `inferencia_ts.py`: se um sistema for adicionado, ambos consomem a mesma lista
e o checkpoint salvo no treino fica consistente com a inferência.
"""

from __future__ import annotations

# Sistemas com dados completos (2000-2026) — chave: id da API; valor: nome canônico.
SISTEMAS: dict[str, str] = {
    "el_64_cantareira":   "Cantareira",
    "el_65_alto_tiete":   "Alto Tietê",
    "el_66_guarapiranga": "Guarapiranga",
    "el_67_cotia":        "Cotia",
    "el_68_rio_grande":   "Rio Grande",
    "el_69_rio_claro":    "Rio Claro",
}

# Parâmetros hidrológicos a usar como features.
# Chave: nome do arquivo JSON; valor: nome curto exposto no DataFrame.
PARAMS: dict[str, str] = {
    "param_1_chuva":       "chuva",
    "param_684_v_util":    "v_util",
    "param_690_q_natural": "q_natural",
    "param_691_q_afluente":"q_afluente",
    "param_687_q_captada": "q_captada",
    "param_689_q_jusante": "q_jusante",
}

# Variável-alvo do modelo.
TARGET_PARAM = "v_util"

# Janela temporal — usada pelo dataset e pela inferência.
LOOKBACK = 90    # dias de histórico que entram no modelo
HORIZON  = 30    # dias previstos pela cabeça do modelo

# Splits temporais (data inclusiva no fim de cada conjunto).
DATA_TREINO_FIM = "2022-12-31"
DATA_VAL_FIM    = "2023-12-31"
# teste: 2024-01-01 em diante
