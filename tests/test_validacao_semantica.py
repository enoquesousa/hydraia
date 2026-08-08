"""Testes da validacao semantica e deteccao de alucinacao."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.validacao_semantica import (
    avaliar_analise,
    convergencia_semantica,
    detectar_alucinacao_textual,
    validar_fisica_hidro,
    validar_fisica_meteo,
    validar_schema_hidro,
    validar_schema_meteo,
)


# ─── Schema hidro ────────────────────────────────────────────────────────────

class TestSchemaHidro:
    def test_aceita_completo(self):
        d = {"nivel_risco": 3, "tendencia": "estavel", "analise": "x", "confianca_pct": 80}
        r = validar_schema_hidro(d)
        assert r.ok

    def test_falta_campo(self):
        d = {"nivel_risco": 3, "tendencia": "estavel", "analise": "x"}  # falta confianca
        r = validar_schema_hidro(d)
        assert not r.ok
        assert "campos faltando" in r.motivos[0]

    def test_tendencia_invalida(self):
        d = {"nivel_risco": 3, "tendencia": "subindo",  # invalido
             "analise": "x", "confianca_pct": 50}
        r = validar_schema_hidro(d)
        assert not r.ok

    def test_sistemas_criticos_nao_lista(self):
        d = {"nivel_risco": 3, "tendencia": "estavel", "analise": "x",
             "confianca_pct": 80, "sistemas_criticos": "Cantareira"}  # string, deveria ser lista
        r = validar_schema_hidro(d)
        assert not r.ok


# ─── Schema meteo ────────────────────────────────────────────────────────────

class TestSchemaMeteo:
    def test_aceita_completo(self):
        d = {
            "nivel_risco": 2,
            "chuva_total_esperada_mm": 30.0,
            "impacto_nos_reservatorios": "positivo",
            "dias_com_chuva_significativa": 3,
            "analise": "x",
            "confianca_pct": 70,
        }
        r = validar_schema_meteo(d)
        assert r.ok

    def test_impacto_invalido(self):
        d = {
            "nivel_risco": 2,
            "chuva_total_esperada_mm": 30.0,
            "impacto_nos_reservatorios": "muito_positivo",  # invalido
            "dias_com_chuva_significativa": 3,
            "analise": "x",
            "confianca_pct": 70,
        }
        r = validar_schema_meteo(d)
        assert not r.ok


# ─── Fisica meteo ────────────────────────────────────────────────────────────

class TestFisicaMeteo:
    def test_chuva_negativa_alerta(self):
        r = validar_fisica_meteo({"chuva_total_esperada_mm": -10})
        assert not r.ok

    def test_chuva_absurda_alerta(self):
        r = validar_fisica_meteo({"chuva_total_esperada_mm": 5000})
        assert not r.ok

    def test_dias_invalidos(self):
        r = validar_fisica_meteo({
            "chuva_total_esperada_mm": 30, "dias_com_chuva_significativa": 999,
        })
        assert not r.ok


# ─── Fisica hidro ────────────────────────────────────────────────────────────

class TestFisicaHidro:
    def test_volume_alto_e_risco_critico_e_incoerente(self):
        d = {"nivel_risco": 5}
        r = validar_fisica_hidro(d, volume_pct=85.0)
        assert not r.ok

    def test_volume_baixo_e_risco_minimo_e_incoerente(self):
        d = {"nivel_risco": 1}
        r = validar_fisica_hidro(d, volume_pct=15.0)
        assert not r.ok

    def test_combinacao_coerente_passa(self):
        d = {"nivel_risco": 3, "tendencia": "estavel"}
        r = validar_fisica_hidro(d, volume_pct=50.0)
        assert r.ok

    def test_melhora_com_risco_5_e_incoerente(self):
        r = validar_fisica_hidro({"nivel_risco": 5, "tendencia": "melhora"})
        assert not r.ok


# ─── Alucinacao textual ─────────────────────────────────────────────────────

class TestAlucinacaoTextual:
    def test_detecta_simulei(self):
        r = detectar_alucinacao_textual({"analise": "Eu simulei o cenario..."})
        assert not r.ok

    def test_detecta_inventei(self):
        r = detectar_alucinacao_textual({"analise": "inventei numeros"})
        assert not r.ok

    def test_detecta_nao_tenho_dados(self):
        r = detectar_alucinacao_textual({"analise": "Nao tenho dados sobre isso."})
        assert not r.ok

    def test_texto_normal_passa(self):
        r = detectar_alucinacao_textual({
            "analise": "Volume estavel, sem riscos relevantes nos proximos 7 dias.",
        })
        assert r.ok


# ─── Convergencia semantica ──────────────────────────────────────────────────

class TestConvergenciaSemantica:
    def test_concordancia_total(self):
        h = {"nivel_risco": 3, "tendencia": "estavel", "confianca_pct": 80}
        m = {"nivel_risco": 3, "impacto_nos_reservatorios": "neutro", "confianca_pct": 80}
        score, _ = convergencia_semantica(h, m)
        assert score > 0.9

    def test_discordancia_total(self):
        h = {"nivel_risco": 1, "tendencia": "melhora", "confianca_pct": 20}
        m = {"nivel_risco": 5, "impacto_nos_reservatorios": "negativo", "confianca_pct": 100}
        score, _ = convergencia_semantica(h, m)
        assert score < 0.3

    def test_um_ausente(self):
        score, obs = convergencia_semantica({"nivel_risco": 3}, None)
        assert score == 0.0


# ─── Pipeline integrado ──────────────────────────────────────────────────────

class TestAvaliarAnalise:
    def test_analise_ausente_devolve_alerta(self):
        d, alertas = avaliar_analise("X", None, tipo="hidro")
        assert d is None
        assert alertas

    def test_alertas_acumulados(self):
        d = {"nivel_risco": 5, "tendencia": "melhora",
             "analise": "Eu simulei isso", "confianca_pct": 80}
        out, alertas = avaliar_analise("X", d, tipo="hidro", volume_pct=80.0)
        assert out is None  # pista textual de alucinacao e fatal por padrao
        # deve conter: incoerencia volume vs risco, tendencia=melhora+risco=5, alucinacao
        assert any("fisica" in a for a in alertas)
        assert any("alucinacao" in a for a in alertas)
