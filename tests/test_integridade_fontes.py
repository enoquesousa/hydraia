"""Garantias de proveniencia para a execucao estritamente real."""

from unittest.mock import patch

from src.core import integracao


def test_slug_sistema_remove_acentos():
    assert integracao._slug_sistema("Alto Tietê") == "alto_tiete"
    assert integracao._slug_sistema("São Lourenço") == "sao_lourenco"


def test_execucao_real_nao_reutiliza_snapshot_em_cache(monkeypatch):
    snapshot = {
        "modo": "snapshot",
        "sistemas": {"cantareira": {"volume_percentual": 33.3}},
    }
    coleta_real = {
        "modo": "real",
        "sistemas": {"cantareira": {"volume_percentual": 34.0}},
    }
    monkeypatch.setattr(integracao._cache_sabesp, "get", lambda _: snapshot)
    monkeypatch.setattr(integracao._cache_sabesp, "set", lambda *_: True)

    with patch(
        "src.api.sabesp.SabespCollector.coletar_macrometropole_completa",
        return_value=coleta_real,
    ) as coletar:
        resultado = integracao.coletar_dados_sabesp_reais(allow_snapshot=False)

    assert resultado["modo"] == "real"
    coletar.assert_called_once_with(allow_snapshot=False)


def test_execucao_offline_pode_reutilizar_snapshot_em_cache(monkeypatch):
    snapshot = {
        "modo": "snapshot",
        "sistemas": {"cantareira": {"volume_percentual": 33.3}},
    }
    monkeypatch.setattr(integracao._cache_sabesp, "get", lambda _: snapshot)

    assert integracao.coletar_dados_sabesp_reais(allow_snapshot=True) is snapshot
