"""Testes do fallback de previsao quando o HydroLSTM nao esta disponivel."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core import integracao


# ─── _baseline_previsoes ────────────────────────────────────────────────────

class TestBaselinePersistence:
    def test_gera_horizonte_correto(self):
        out = integracao._baseline_previsoes(7)
        assert len(out) == 7

    def test_marca_fonte(self):
        out = integracao._baseline_previsoes(3)
        assert all(p["fonte_previsao"] == "baseline_persistence" for p in out)

    def test_inclui_todos_os_sistemas(self):
        out = integracao._baseline_previsoes(1)
        chaves = out[0].keys()
        assert "volume_cantareira_pct" in chaves
        assert "volume_alto_tiete_pct" in chaves
        assert "volume_guarapiranga_pct" in chaves

    def test_dia_e_int_a_partir_de_1(self):
        out = integracao._baseline_previsoes(3)
        assert out[0]["dia"] == 1.0
        assert out[2]["dia"] == 3.0

    def test_persistencia_usa_dados_da_execucao(self):
        dados = {
            "sistemas": {
                "cantareira": {"volume_percentual": 47.25},
                "alto_tiete": {"volume_percentual": 55.0},
            }
        }
        out = integracao._baseline_previsoes(2, dados_sabesp=dados)
        assert all(p["volume_cantareira_pct"] == 47.25 for p in out)


# ─── gerar_previsoes_ml com fallback ────────────────────────────────────────

class TestGerarPrevisoesFallback:
    def test_modo_simulado_dados_usa_baseline_direto(self):
        out = integracao.gerar_previsoes_ml(horizonte=5, modo_simulado_dados=True)
        assert len(out) == 5
        assert out[0]["fonte_previsao"] == "baseline_persistence"

    def test_lstm_falha_e_fallback_permitido_usa_baseline(self):
        with patch("src.data.inferencia_ts.prever", side_effect=RuntimeError("sem modelo")):
            out = integracao.gerar_previsoes_ml(
                horizonte=4, permitir_baseline_fallback=True,
            )
        assert len(out) == 4
        assert out[0]["fonte_previsao"].startswith("baseline_")

    def test_lstm_falha_e_fallback_negado_propaga(self):
        with patch("src.data.inferencia_ts.prever", side_effect=RuntimeError("sem modelo")):
            with pytest.raises(RuntimeError, match="sem modelo"):
                integracao.gerar_previsoes_ml(
                    horizonte=4, permitir_baseline_fallback=False,
                )


# ─── Pipeline integrado com fallback (skip_llm) ─────────────────────────────

class TestPipelineComFallback:
    def test_skip_llm_e_modo_simulado_funciona_sem_torch(self, tmp_path):
        """Cenario TCC: demo offline sem chave, sem torch, sem LM Studio."""
        # Cria o arquivo de contexto fake
        md = tmp_path / "ctx.md"
        md.write_text("contexto cientifico minimo", encoding="utf-8")

        relatorio = integracao.executar_pipeline_integrado(
            caminho_md=md,
            horizonte_dias=3,
            modo_simulado_dados=True,
            allow_snapshot=False,
            skip_llm=True,
            permitir_baseline_fallback=True,
        )
        assert "DRY-RUN" in relatorio
        assert "simulado" in relatorio
        assert "LLMs nao foram chamados" in relatorio

    def test_skip_llm_inclui_previsoes_no_relatorio(self, tmp_path):
        md = tmp_path / "ctx.md"
        md.write_text("contexto", encoding="utf-8")
        relatorio = integracao.executar_pipeline_integrado(
            caminho_md=md, horizonte_dias=3,
            modo_simulado_dados=True, skip_llm=True,
        )
        assert "Previsao de volume" in relatorio
        assert "baseline_persistence" in relatorio
        assert "Dia +1:" in relatorio
