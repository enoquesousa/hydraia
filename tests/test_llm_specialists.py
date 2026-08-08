"""Testes unitários para Strategy Pattern dos LLM Specialists.

Cobre: extrair_json, validar_analise, retry, fallback de temperature,
short-circuit em chave ausente, propagação de erros permanentes (StopRetry).
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.llm_specialists import (
    GPTSpecialist,
    LLMSpecialist,
    StopRetry,
    extrair_json,
    validar_analise,
)


# ─── Helper: subclasse de teste que conta invocações ────────────────────────

class _FakeSpecialist(LLMSpecialist):
    """Specialist controlável para testes — não fala com nenhum SDK real."""

    def __init__(self, respostas, **kwargs):
        super().__init__(api_key="fake", model="fake-model",
                         max_retries=2, backoff_s=0, **kwargs)
        self.respostas = list(respostas)
        self.invocacoes = 0

    def _invocar(self, prompt: str) -> str:
        self.invocacoes += 1
        item = self.respostas.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ─── extrair_json ────────────────────────────────────────────────────────────

class TestExtrairJsonNovo:
    def test_bloco_markdown(self):
        assert extrair_json('```json\n{"x": 1}\n```') == {"x": 1}

    def test_json_inline(self):
        assert extrair_json('blá {"x": 2} blá')["x"] == 2

    def test_string_vazia_retorna_none(self):
        assert extrair_json("") is None

    def test_none_retorna_none(self):
        assert extrair_json(None) is None

    def test_lixo_retorna_none(self):
        assert extrair_json("texto sem json") is None


# ─── validar_analise ─────────────────────────────────────────────────────────

class TestValidarAnaliseNovo:
    def test_risco_string_numerica_e_aceita(self):
        # int("3") funciona → o validador aceita
        r = validar_analise({"nivel_risco": "3", "confianca_pct": 50}, "T")
        assert r["nivel_risco"] == 3

    def test_confianca_negativa_clamp_zero(self):
        r = validar_analise({"nivel_risco": 2, "confianca_pct": -10}, "T")
        assert r["confianca_pct"] == 0


# ─── Retry behavior ───────────────────────────────────────────────────────────

class TestRetry:
    def test_sucesso_na_primeira(self):
        spec = _FakeSpecialist(['{"nivel_risco": 3, "confianca_pct": 80}'])
        assert spec.analisar("prompt")["nivel_risco"] == 3
        assert spec.invocacoes == 1

    def test_falha_transitoria_com_retry(self):
        spec = _FakeSpecialist([
            ConnectionError("rede caiu"),
            '{"nivel_risco": 4, "confianca_pct": 70}',
        ])
        result = spec.analisar("prompt")
        assert result["nivel_risco"] == 4
        assert spec.invocacoes == 2

    def test_todas_tentativas_falham(self):
        spec = _FakeSpecialist([
            TimeoutError("t1"),
            TimeoutError("t2"),
            TimeoutError("t3"),
        ])
        assert spec.analisar("prompt") is None
        assert spec.invocacoes == 3   # max_retries=2 → 3 chamadas no total

    def test_stop_retry_curtocircuita(self):
        spec = _FakeSpecialist([
            StopRetry("auth invalida"),
            '{"nivel_risco": 1}',  # não deve chegar aqui
        ])
        assert spec.analisar("prompt") is None
        assert spec.invocacoes == 1   # parou na primeira

    def test_chave_ausente_pula_invocacao(self):
        spec = _FakeSpecialist(['{"nivel_risco": 5, "confianca_pct": 60}'])
        spec.api_key = None
        assert spec.analisar("prompt") is None
        assert spec.invocacoes == 0   # short-circuit antes de invocar


# ─── GPT-specific: detecção de modelos sem temperature ──────────────────────

class TestGPTAceitaTemperature:
    @pytest.mark.parametrize("model,esperado", [
        ("gpt-4o", True),
        ("gpt-4-turbo", True),
        ("gpt-3.5-turbo", True),
        ("gpt-5", False),
        ("gpt-5.1-chat-latest", False),
        ("o1-preview", False),
        ("o3-mini", False),
        ("GPT-5", False),         # case insensitive
    ])
    def test_deteccao(self, model, esperado):
        assert GPTSpecialist._aceita_temperature(model) is esperado
