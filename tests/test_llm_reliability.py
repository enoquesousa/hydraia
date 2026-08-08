"""Testes de confiabilidade dos LLM Specialists — timeout, breaker, watchdog, truncagem."""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.llm_specialists import (
    DEFAULT_MAX_PROMPT_CHARS,
    LLMSpecialist,
    StopRetry,
    truncar_prompt,
)
from src.utils.circuit_breaker import CircuitBreaker


# ─── Helper: subclasse fake controlavel ──────────────────────────────────────

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


# ─── Truncagem de prompt ─────────────────────────────────────────────────────

class TestTruncagemPrompt:
    def test_nao_modifica_prompt_pequeno(self):
        p = "abc" * 10
        assert truncar_prompt(p, max_chars=1000) == p

    def test_trunca_prompt_grande(self):
        p = "x" * (DEFAULT_MAX_PROMPT_CHARS + 1000)
        out = truncar_prompt(p)
        assert len(out) <= DEFAULT_MAX_PROMPT_CHARS
        assert "[...conteudo truncado por watchdog...]" in out

    def test_preserva_inicio_e_fim(self):
        head = "INSTRUCOES_INICIAIS_AQUI"
        tail = "INSTRUCOES_FINAIS_JSON_AQUI"
        meio = "y" * (DEFAULT_MAX_PROMPT_CHARS + 1000)
        p = head + meio + tail
        out = truncar_prompt(p)
        assert out.startswith(head)
        assert out.endswith(tail)


# ─── Timeout local ───────────────────────────────────────────────────────────

class TestTimeoutLocal:
    def test_timeout_curto_aborta_chamada(self):
        def lento(_):
            time.sleep(2.0)
            return "tarde"
        spec = _FakeSpec(lento, timeout_s=0.2)
        # analisar() deve retornar None (timeout vira "todas tentativas falharam")
        assert spec.analisar("p") is None

    def test_timeout_nao_dispara_quando_rapido(self):
        def rapido(_):
            return '{"nivel_risco": 2, "confianca_pct": 60}'
        spec = _FakeSpec(rapido, timeout_s=2.0)
        result = spec.analisar("p")
        assert result is not None
        assert result["nivel_risco"] == 2


# ─── Watchdog de resposta ────────────────────────────────────────────────────

class TestWatchdogResposta:
    def test_resposta_gigantesca_e_truncada(self):
        # Resposta com lixo no inicio + JSON valido logo apos
        gigante = "lixo " * 1000 + '{"nivel_risco": 3, "confianca_pct": 50}'
        def fn(_):
            return gigante
        spec = _FakeSpec(fn, max_response_chars=200)
        # Mesmo truncado, o JSON ainda deve ser parseavel se cair na janela
        # — neste caso o JSON esta no final, entao truncar deve eliminar e retornar None
        assert spec.analisar("p") is None

    def test_resposta_dentro_do_limite_e_processada(self):
        def fn(_):
            return '{"nivel_risco": 4, "confianca_pct": 90}'
        spec = _FakeSpec(fn, max_response_chars=100)
        result = spec.analisar("p")
        assert result is not None
        assert result["nivel_risco"] == 4


# ─── Circuit Breaker integrado ───────────────────────────────────────────────

class TestBreakerIntegrado:
    def test_breaker_abre_apos_falhas_e_curto_circuita(self):
        chamadas = []

        def fn(_):
            chamadas.append(1)
            raise RuntimeError("provider caiu")

        spec = _FakeSpec(
            fn,
            max_retries=0,
            backoff_s=0,
            timeout_s=2.0,
            breaker_threshold=2,
            breaker_recovery_s=10.0,
        )
        # 2 falhas -> abre o breaker
        spec.analisar("p")
        spec.analisar("p")
        assert spec.breaker.state == CircuitBreaker.OPEN
        # 3a chamada e curto-circuitada (sem invocar fn)
        n_antes = len(chamadas)
        result = spec.analisar("p")
        assert result is None
        assert len(chamadas) == n_antes  # nao chamou fn de novo


# ─── Curto-circuito quando chave ausente ─────────────────────────────────────

class TestSemChave:
    def test_sem_api_key_pula_invocacao(self):
        spec = _FakeSpec(lambda _: '{"nivel_risco": 1}')
        spec.api_key = None
        assert spec.analisar("p") is None
        assert spec.invocacoes == 0


# ─── StopRetry ainda funciona ────────────────────────────────────────────────

class TestStopRetry:
    def test_stop_retry_curto_circuita(self):
        def fn(_):
            raise StopRetry("auth invalida")
        spec = _FakeSpec(fn, max_retries=2)
        assert spec.analisar("p") is None
        assert spec.invocacoes == 1
