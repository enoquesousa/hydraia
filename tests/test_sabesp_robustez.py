"""Testes de robustez do SabespCollector — schema, snapshot, simulado."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.sabesp import (
    SabespCollector,
    SchemaError,
    _validar_resposta_resumo,
    _validar_resposta_ultima_data,
)


# ─── Schema validators ───────────────────────────────────────────────────────

class TestValidarUltimaData:
    def test_aceita_payload_valido(self):
        out = _validar_resposta_ultima_data({"succeeded": True, "data": "2026-04-14"})
        assert out == "2026-04-14"

    def test_rejeita_succeeded_false(self):
        with pytest.raises(SchemaError):
            _validar_resposta_ultima_data({"succeeded": False, "data": "2026-04-14"})

    def test_rejeita_data_invalida(self):
        with pytest.raises(SchemaError):
            _validar_resposta_ultima_data({"succeeded": True, "data": "data-errada"})

    def test_rejeita_sem_data(self):
        with pytest.raises(SchemaError):
            _validar_resposta_ultima_data({"succeeded": True})

    def test_rejeita_nao_dict(self):
        with pytest.raises(SchemaError):
            _validar_resposta_ultima_data([])


class TestValidarResumo:
    def test_aceita_lista_valida(self):
        out = _validar_resposta_resumo({
            "data": [
                {"idSistema": 64, "volumeUtilArmazenadoPorcentagem": 50.0},
            ]
        })
        assert len(out) == 1

    def test_rejeita_data_vazio(self):
        with pytest.raises(SchemaError):
            _validar_resposta_resumo({"data": []})

    def test_ignora_itens_invalidos(self):
        out = _validar_resposta_resumo({
            "data": [
                {"idSistema": 64, "volumeUtilArmazenadoPorcentagem": 50.0},
                {"idSistema": None, "volumeUtilArmazenadoPorcentagem": None},  # invalido
                "string_qualquer",  # tipo errado
            ]
        })
        assert len(out) == 1

    def test_normaliza_string_numerica(self):
        out = _validar_resposta_resumo({
            "data": [{
                "idSistema": 64,
                "volumeUtilArmazenadoPorcentagem": 50.0,
                "chuva": "12,5",  # virgula decimal
            }]
        })
        assert out[0]["chuva"] == 12.5

    def test_rejeita_todos_invalidos(self):
        with pytest.raises(SchemaError):
            _validar_resposta_resumo({"data": [
                {"idSistema": None, "volumeUtilArmazenadoPorcentagem": None},
            ]})


# ─── Snapshot fallback ───────────────────────────────────────────────────────

class TestSnapshot:
    def test_sem_snapshot_e_api_off_falha(self, tmp_path):
        col = SabespCollector(data_dir=tmp_path, auto_save=False, retry_attempts=1)
        with patch("src.api.sabesp.requests.get", side_effect=Exception("offline")):
            with pytest.raises(RuntimeError):
                col.coletar_macrometropole_completa(allow_snapshot=True)

    def test_snapshot_e_carregado_em_falha(self, tmp_path):
        snapshot = {
            "fonte": "SABESP (Oficial)",
            "data_referencia": "2026-04-10",
            "sistemas": {"cantareira": {"volume_percentual": 55.0}},
            "resumo": {"total_sistemas": 1, "volume_medio_percentual": 55.0},
        }
        (tmp_path / "ultimo_snapshot_valido.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )
        col = SabespCollector(data_dir=tmp_path, auto_save=False, retry_attempts=1)
        with patch("src.api.sabesp.requests.get", side_effect=Exception("offline")):
            dados = col.coletar_macrometropole_completa(allow_snapshot=True)
        assert dados["modo"] == "snapshot"
        assert "fallback_motivo" in dados

    def test_allow_snapshot_false_falha_mesmo_com_snapshot(self, tmp_path):
        snapshot = {"sistemas": {"x": {"volume_percentual": 1}}}
        (tmp_path / "ultimo_snapshot_valido.json").write_text(json.dumps(snapshot), encoding="utf-8")
        col = SabespCollector(data_dir=tmp_path, auto_save=False, retry_attempts=1)
        with patch("src.api.sabesp.requests.get", side_effect=Exception("offline")):
            with pytest.raises(RuntimeError):
                col.coletar_macrometropole_completa(allow_snapshot=False)

    def test_snapshot_e_atualizado_em_sucesso(self, tmp_path):
        col = SabespCollector(data_dir=tmp_path, auto_save=True, retry_attempts=1)

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "ultima-data" in url:
                resp.json.return_value = {"succeeded": True, "data": "2026-04-14"}
            else:
                resp.json.return_value = {
                    "data": [{
                        "idSistema": 64,
                        "volumeUtilArmazenadoPorcentagem": 60.0,
                        "chuva": 1.0,
                    }]
                }
            return resp

        with patch("src.api.sabesp.requests.get", side_effect=side_effect):
            col.coletar_macrometropole_completa()

        assert (tmp_path / "ultimo_snapshot_valido.json").exists()


# ─── Modo simulado ───────────────────────────────────────────────────────────

class TestModoSimulado:
    def test_retorna_dados_validos(self):
        col = SabespCollector(auto_save=False)
        dados = col.coletar_simulado()
        assert dados["modo"] == "simulado"
        assert "cantareira" in dados["sistemas"]
        assert isinstance(dados["sistemas"]["cantareira"]["volume_percentual"], float)
        assert dados["resumo"]["total_sistemas"] == 7

    def test_data_customizada(self):
        col = SabespCollector(auto_save=False)
        dados = col.coletar_simulado(data_ref="2026-01-01")
        assert dados["data_referencia"] == "2026-01-01"


# ─── Circuit breaker ─────────────────────────────────────────────────────────

class TestCircuitBreakerSabesp:
    def test_breaker_abre_apos_falhas(self, tmp_path):
        col = SabespCollector(
            data_dir=tmp_path, auto_save=False, retry_attempts=1,
            breaker_threshold=2, breaker_recovery_s=60,
        )
        with patch("src.api.sabesp.requests.get", side_effect=Exception("net")):
            # 2 falhas — breaker abre
            col._fazer_requisicao("/x")
            col._fazer_requisicao("/y")
        from src.utils.circuit_breaker import CircuitBreaker as CB
        assert col._breaker.state == CB.OPEN
