"""Testes do modo strict_schema do LLMSpecialist."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.llm_specialists import LLMSpecialist
from src.core.validacao_semantica import CAMPOS_HIDRO, CAMPOS_METEO
from src.utils.budget import BUDGET, BudgetLimits


class _FakeSpec(LLMSpecialist):
    def __init__(self, fn, **kw):
        kw.setdefault("max_retries", 0)
        kw.setdefault("backoff_s", 0)
        super().__init__(api_key="fake", model="fake", **kw)
        self._fn = fn

    def _invocar(self, prompt: str) -> str:
        return self._fn(prompt)


@pytest.fixture(autouse=True)
def _reset_budget():
    BUDGET.reset()
    BUDGET.limites = BudgetLimits()
    yield
    BUDGET.reset()
    BUDGET.limites = BudgetLimits()


# ─── strict_schema ativo ─────────────────────────────────────────────────────

class TestStrictSchema:
    def test_aceita_resposta_completa(self):
        resp = (
            '{"nivel_risco": 3, "tendencia": "estavel", "analise": "ok",'
            ' "confianca_pct": 80}'
        )
        spec = _FakeSpec(
            lambda p: resp,
            strict_schema=True,
            campos_obrigatorios=CAMPOS_HIDRO,
        )
        out = spec.analisar("um prompt de tamanho razoavel para gerar tokens")
        assert out is not None
        assert out["nivel_risco"] == 3

    def test_descarta_resposta_sem_tendencia(self):
        # Falta tendencia (campo obrigatorio em CAMPOS_HIDRO)
        resp = '{"nivel_risco": 3, "analise": "ok", "confianca_pct": 80}'
        spec = _FakeSpec(
            lambda p: resp,
            strict_schema=True,
            campos_obrigatorios=CAMPOS_HIDRO,
        )
        assert spec.analisar("prompt grande o suficiente") is None

    def test_descarta_resposta_meteo_sem_impacto(self):
        # Falta impacto_nos_reservatorios
        resp = (
            '{"nivel_risco": 2, "chuva_total_esperada_mm": 10.0, '
            '"dias_com_chuva_significativa": 2, "analise": "x", "confianca_pct": 60}'
        )
        spec = _FakeSpec(
            lambda p: resp,
            strict_schema=True,
            campos_obrigatorios=CAMPOS_METEO,
        )
        assert spec.analisar("prompt razoavel") is None


# ─── strict_schema desligado (compat) ────────────────────────────────────────

class TestStrictSchemaDesligado:
    def test_aceita_resposta_incompleta_quando_off(self):
        resp = '{"nivel_risco": 3, "confianca_pct": 80}'
        spec = _FakeSpec(
            lambda p: resp,
            strict_schema=False,                     # default
            campos_obrigatorios=CAMPOS_HIDRO,
        )
        out = spec.analisar("prompt com tamanho ok")
        assert out is not None
        assert out["nivel_risco"] == 3
