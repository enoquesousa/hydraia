"""Testes do CircuitBreaker — transicoes de estado e thread-safety basico."""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError


def _falha():
    raise RuntimeError("boom")


def _sucesso():
    return 42


class TestEstadoInicial:
    def test_comeca_fechado(self):
        cb = CircuitBreaker("t")
        assert cb.state == CircuitBreaker.CLOSED

    def test_passa_chamada_quando_fechado(self):
        cb = CircuitBreaker("t")
        assert cb.call(_sucesso) == 42


class TestAbertura:
    def test_abre_apos_threshold(self):
        cb = CircuitBreaker("t", failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_falha)
        assert cb.state == CircuitBreaker.OPEN

    def test_curto_circuita_quando_aberto(self):
        cb = CircuitBreaker("t", failure_threshold=2)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(_falha)
        # Agora aberto: nao deve nem chamar fn
        with pytest.raises(CircuitOpenError):
            cb.call(_sucesso)


class TestRecuperacao:
    def test_half_open_apos_recovery(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_s=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_falha)
        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.06)
        # Acessar state dispara transicao para HALF_OPEN
        assert cb.state == CircuitBreaker.HALF_OPEN

    def test_half_open_sucesso_fecha(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_s=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_falha)
        time.sleep(0.06)
        assert cb.call(_sucesso) == 42
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_falha_reabre(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_s=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_falha)
        time.sleep(0.06)
        with pytest.raises(RuntimeError):
            cb.call(_falha)
        assert cb.state == CircuitBreaker.OPEN


class TestReset:
    def test_reset_volta_para_closed(self):
        cb = CircuitBreaker("t", failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_falha)
        assert cb.state == CircuitBreaker.OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.call(_sucesso) == 42
