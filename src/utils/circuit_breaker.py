"""Circuit Breaker simples e thread-safe.

Quando um serviço externo (Gemini, GPT, OpenWeather, SABESP, LM Studio)
falha N vezes consecutivas, o circuito abre e bloqueia novas chamadas por
`recovery_s` segundos — evita gastar tokens e tempo em providers já caídos.

Estados:
    CLOSED   — operação normal; falhas são contadas.
    OPEN     — todas as chamadas são curto-circuitadas (CircuitOpenError).
    HALF_OPEN — após `recovery_s`, deixa passar UMA chamada de teste; se ela
                tem sucesso, fecha de novo; se falha, reabre.

Por que aqui e não numa lib? Para não adicionar dependência externa pelo TCC,
e o comportamento que precisamos é mínimo (3 estados, contagem simples).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Levantado quando o circuito está aberto e bloqueia a chamada."""


class CircuitBreaker:
    """Circuit breaker thread-safe com 3 estados."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        nome: str,
        failure_threshold: int = 3,
        recovery_s: float = 60.0,
    ) -> None:
        self.nome = nome
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def reset(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0
            self._opened_at = None

    def _maybe_half_open_locked(self) -> None:
        if self._state == self.OPEN and self._opened_at is not None:
            if (time.time() - self._opened_at) >= self.recovery_s:
                logger.info("[CircuitBreaker:%s] OPEN → HALF_OPEN (timeout de recuperacao)", self.nome)
                self._state = self.HALF_OPEN

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Executa `fn(*args, **kwargs)`. Lança CircuitOpenError se aberto."""
        with self._lock:
            self._maybe_half_open_locked()
            if self._state == self.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.nome}' aberto — bloqueando chamada"
                )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                logger.info("[CircuitBreaker:%s] HALF_OPEN → CLOSED (recuperado)", self.nome)
            self._state = self.CLOSED
            self._failures = 0
            self._opened_at = None

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == self.HALF_OPEN:
                logger.warning("[CircuitBreaker:%s] HALF_OPEN → OPEN (falha em chamada de teste)", self.nome)
                self._state = self.OPEN
                self._opened_at = time.time()
                return
            if self._failures >= self.failure_threshold:
                logger.warning(
                    "[CircuitBreaker:%s] CLOSED → OPEN (%d falhas consecutivas, recuperar em %.0fs)",
                    self.nome, self._failures, self.recovery_s,
                )
                self._state = self.OPEN
                self._opened_at = time.time()
