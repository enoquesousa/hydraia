import math

import numpy as np
import pytest

from src.data.baselines import prob_evento
from src.data.estatisticas_prob import comparar_valor_atual, estatisticas_mensais


def test_prob_evento_monotonic():
    residuos = np.array([-2.0, 0.0, 2.0])
    # Centro 50, limite 49 → 1/3 dos resíduos deixam abaixo
    assert prob_evento(50.0, residuos, 49.0, direcao="abaixo") == pytest.approx(1 / 3, rel=1e-3)
    # Direção acima
    assert prob_evento(50.0, residuos, 51.0, direcao="acima") == pytest.approx(1 / 3, rel=1e-3)


def test_comparar_valor_usa_climatologia():
    stats = estatisticas_mensais()
    if not stats:
        pytest.skip("Sem séries históricas para estatísticas mensais")
    sistema = next(iter(stats))
    mes = next(iter(stats[sistema]))
    base = stats[sistema][mes]
    comp = comparar_valor_atual(sistema, base["p50"], mes=mes)
    assert comp is not None
    assert math.isfinite(comp["zscore"])
    assert comp["mes_ref"] == mes
