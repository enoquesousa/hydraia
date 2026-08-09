"""Renderizacao factual do relatorio sem delegar numeros a uma LLM."""

from src.core.orquestrador import _montar_relatorio_final, _secoes_fatuais


def _estado():
    return {
        "data_referencia": "2026-08-08",
        "volume_cantareira_pct": 35.5,
        "macrometropole": {"volume_medio_pct": 50.0, "sistemas_criticos": 0},
        "sistemas": {
            "cantareira": {
                "sistema": "Sistema Cantareira",
                "volume_percentual": 35.5,
                "chuva_mm": 1.2,
                "vazao_natural_m3s": 20.0,
            },
        },
        "anomalias_historicas": {"Cantareira": {"zscore": -0.5}},
        "prob_hist_futuro": {
            "Cantareira": {
                1: {"p50": 40.0},
                30: {"p10": 10.0, "p50": 30.0, "p90": 60.0},
                90: {"p10": 5.0, "p50": 25.0, "p90": 55.0},
            },
        },
    }


def _previsoes():
    return [{
        "dia": 1.0,
        "volume_cantareira_pct": 38.0,
        "volume_cantareira_p10": 30.0,
        "volume_cantareira_p50": 40.0,
        "volume_cantareira_p90": 50.0,
        "prob_cantareira_sub_30_pct": 1.0,
        "prob_cantareira_sub_50_pct": 90.0,
        "prob_cantareira_acima_80_pct": 0.0,
        "volume_alto_tiete_pct": 50.0,
        "volume_alto_tiete_pct_p10": 45.0,
        "volume_alto_tiete_pct_p50": 50.0,
        "volume_alto_tiete_pct_p90": 55.0,
        "volume_alto_tiete_pct_prob_sub_30_pct": 0.0,
        "volume_alto_tiete_pct_prob_sub_50_pct": 50.0,
        "volume_alto_tiete_pct_prob_acima_80_pct": 0.0,
    }]


def test_secoes_fatuais_preservam_sistemas_e_quantis():
    texto = _secoes_fatuais(_estado(), _previsoes(), None)

    assert "| Cantareira | 35.50% | -4.50%" in texto
    assert "| +1 | Cantareira | 30.0% | 40.0% | 50.0%" in texto
    assert "| +1 | Alto Tietê | 45.0% | 50.0% | 55.0%" in texto
    assert "não calibradas" in texto


def test_relatorio_final_exige_secoes_narrativas():
    narrativa = "### 6) Riscos\nOK\n### 7) Recomendações\nOK\n### 8) Insights\nOK"
    texto = _montar_relatorio_final(
        narrativa, _estado(), _previsoes(), None, exigir_secoes=True,
    )

    assert texto.startswith("# RELATÓRIO TÉCNICO FINAL")
    assert texto.count("### 6)") == 1
