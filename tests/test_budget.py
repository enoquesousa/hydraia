"""Testes do BudgetController e integracao com LLMSpecialist."""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.llm_specialists import LLMSpecialist
from src.utils.budget import BUDGET, BudgetController, BudgetExceeded, BudgetLimits


class _FakeSpec(LLMSpecialist):
    def __init__(self, fn, **kw):
        kw.setdefault("max_retries", 0)
        kw.setdefault("backoff_s", 0)
        super().__init__(api_key="fake", model="fake", **kw)
        self._fn = fn
        self.invocacoes = 0

    def _invocar(self, prompt: str) -> str:
        self.invocacoes += 1
        return self._fn(prompt)


# ─── BudgetController unitario ───────────────────────────────────────────────

class TestBudgetUnitario:
    def test_estado_inicial_zerado(self):
        c = BudgetController()
        s = c.snapshot()["estado"]
        assert s["llm_calls"] == 0
        assert s["input_tokens"] == 0

    def test_account_acumula(self):
        c = BudgetController()
        c.account(prompt="abcd" * 10, resposta="xyz" * 5)
        s = c.snapshot()["estado"]
        assert s["llm_calls"] == 1
        assert s["input_tokens"] > 0
        assert s["output_tokens"] > 0

    def test_max_llm_calls_bloqueia(self):
        c = BudgetController(BudgetLimits(max_llm_calls=2))
        c.account(prompt="x", resposta="y")
        c.account(prompt="x", resposta="y")
        with pytest.raises(BudgetExceeded):
            c.account(prompt="x", resposta="y")

    def test_can_call_proativo(self):
        c = BudgetController(BudgetLimits(max_llm_calls=1))
        ok, _ = c.can_call()
        assert ok
        c.account(prompt="x", resposta="y")
        ok, motivo = c.can_call()
        assert not ok
        assert "max_llm_calls" in motivo

    def test_max_input_tokens(self):
        c = BudgetController(BudgetLimits(max_input_tokens=10))
        # ~ 100 chars / 4 = 25 tokens > 10
        ok, _ = c.can_call(prompt="x" * 100)
        assert not ok

    def test_max_wallclock(self):
        c = BudgetController(BudgetLimits(max_wallclock_s=0.05))
        time.sleep(0.06)
        with pytest.raises(BudgetExceeded):
            c.account(prompt="a", resposta="b")

    def test_modo_economico_aplica_limites(self):
        c = BudgetController()
        c.aplicar_modo_economico()
        assert c.limites.max_llm_calls == 6
        assert c.limites.max_wallclock_s == 300

    def test_reset_zera(self):
        c = BudgetController()
        c.account(prompt="x", resposta="y")
        c.reset()
        assert c.snapshot()["estado"]["llm_calls"] == 0


# ─── Integracao com LLMSpecialist ────────────────────────────────────────────

class TestBudgetComLLM:
    def setup_method(self):
        BUDGET.reset()
        BUDGET.limites = BudgetLimits()  # sem limites por default

    def teardown_method(self):
        BUDGET.reset()
        BUDGET.limites = BudgetLimits()

    def test_chamada_normal_contabiliza(self):
        spec = _FakeSpec(lambda p: '{"nivel_risco": 3, "confianca_pct": 80}')
        # prompt grande o suficiente para gerar tokens > 0 (chars/4)
        spec.analisar("prompt qualquer com tamanho razoavel para gerar tokens estimados")
        s = BUDGET.snapshot()["estado"]
        assert s["llm_calls"] == 1
        assert s["input_tokens"] > 0
        assert s["output_tokens"] > 0

    def test_max_calls_excedido_bloqueia_proxima(self):
        BUDGET.limites = BudgetLimits(max_llm_calls=1)
        spec = _FakeSpec(lambda p: '{"nivel_risco": 3}')
        # primeira passa
        spec.analisar("p1")
        # segunda — deve ser bloqueada (can_call retorna False)
        n_antes = spec.invocacoes
        spec.analisar("p2")
        assert spec.invocacoes == n_antes  # nao chamou _invocar

    def test_modo_economico_limita_pipeline(self):
        BUDGET.aplicar_modo_economico()
        spec = _FakeSpec(lambda p: '{"nivel_risco": 2}')
        # 6 chamadas devem passar
        for _ in range(6):
            spec.analisar("p")
        # 7a deve ser bloqueada
        n_antes = spec.invocacoes
        spec.analisar("p")
        assert spec.invocacoes == n_antes
