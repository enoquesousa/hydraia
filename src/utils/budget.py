"""BudgetController — orcamento global de tempo, chamadas e tokens por execucao.

Para o TCC, controla quanto se gasta em uma unica execucao do pipeline:

  * `max_wallclock_s` — tempo total de execucao
  * `max_llm_calls` — numero total de chamadas LLM
  * `max_input_tokens_estimados` — soma de chars/4 dos prompts
  * `max_output_tokens_estimados` — soma de chars/4 das respostas

Cada limite, quando excedido, levanta `BudgetExceeded`. Componentes podem
checar `controller.can_call()` antes de fazer a chamada (modo proativo) ou
deixar o controller agir reativamente via `controller.account(...)`.

Tambem expoe um "modo economico" — preset que reduz max_tokens e max_retries
de cada especialista para minimizar custo numa demonstracao.

Os tokens sao estimados em chars/4 (regra-de-bolso para inglês/portugues).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Levantado quando algum limite do orcamento e violado."""


@dataclass
class BudgetLimits:
    max_wallclock_s: Optional[float] = None
    max_llm_calls: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None


@dataclass
class BudgetState:
    started_at: float = field(default_factory=time.time)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _estimar_tokens(texto: str) -> int:
    return max(0, len(texto or "") // 4)


class BudgetController:
    """Thread-safe. Mantem estado e limites; oferece checagem proativa e reativa."""

    def __init__(self, limites: Optional[BudgetLimits] = None) -> None:
        self.limites = limites or BudgetLimits()
        self.estado = BudgetState()
        self._lock = threading.RLock()

    # ── Configuracao ──

    def reset(self) -> None:
        with self._lock:
            self.estado = BudgetState()

    def aplicar_modo_economico(self) -> None:
        """Preset suave para demos: 5min, 6 chamadas LLM, ~10k tokens entrada/saida."""
        self.limites = BudgetLimits(
            max_wallclock_s=300.0,
            max_llm_calls=6,
            max_input_tokens=10_000,
            max_output_tokens=10_000,
        )

    # ── Checagem proativa ──

    def can_call(self, *, prompt: str = "", custo_min_calls: int = 1) -> tuple[bool, str]:
        with self._lock:
            self._verificar_wallclock(strict=False)
            lim = self.limites
            est = self.estado
            entrada = _estimar_tokens(prompt)

            if lim.max_llm_calls is not None and (est.llm_calls + custo_min_calls) > lim.max_llm_calls:
                return False, f"max_llm_calls ({lim.max_llm_calls}) seria excedido"
            if lim.max_input_tokens is not None and (est.input_tokens + entrada) > lim.max_input_tokens:
                return False, f"max_input_tokens ({lim.max_input_tokens}) seria excedido"
            if lim.max_wallclock_s is not None:
                if (time.time() - est.started_at) > lim.max_wallclock_s:
                    return False, f"max_wallclock_s ({lim.max_wallclock_s}s) ja excedido"
            return True, "ok"

    # ── Contabilidade reativa ──

    def account(
        self,
        *,
        prompt: str = "",
        resposta: str = "",
        n_calls: int = 1,
        strict: bool = True,
    ) -> None:
        with self._lock:
            self.estado.llm_calls += n_calls
            self.estado.input_tokens += _estimar_tokens(prompt)
            self.estado.output_tokens += _estimar_tokens(resposta)
            if strict:
                self._enforcar()

    def _enforcar(self) -> None:
        """Levanta BudgetExceeded se algum limite ja foi violado."""
        lim = self.limites
        est = self.estado
        if lim.max_wallclock_s is not None and (time.time() - est.started_at) > lim.max_wallclock_s:
            raise BudgetExceeded(f"wallclock {lim.max_wallclock_s}s excedido")
        if lim.max_llm_calls is not None and est.llm_calls > lim.max_llm_calls:
            raise BudgetExceeded(f"max_llm_calls {lim.max_llm_calls} excedido")
        if lim.max_input_tokens is not None and est.input_tokens > lim.max_input_tokens:
            raise BudgetExceeded(f"max_input_tokens {lim.max_input_tokens} excedido")
        if lim.max_output_tokens is not None and est.output_tokens > lim.max_output_tokens:
            raise BudgetExceeded(f"max_output_tokens {lim.max_output_tokens} excedido")

    def _verificar_wallclock(self, strict: bool = True) -> None:
        lim = self.limites
        est = self.estado
        if lim.max_wallclock_s is None:
            return
        if (time.time() - est.started_at) > lim.max_wallclock_s and strict:
            raise BudgetExceeded(f"wallclock {lim.max_wallclock_s}s excedido")

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "limites": {
                    "max_wallclock_s": self.limites.max_wallclock_s,
                    "max_llm_calls": self.limites.max_llm_calls,
                    "max_input_tokens": self.limites.max_input_tokens,
                    "max_output_tokens": self.limites.max_output_tokens,
                },
                "estado": {
                    "elapsed_s": round(time.time() - self.estado.started_at, 2),
                    "llm_calls": self.estado.llm_calls,
                    "input_tokens": self.estado.input_tokens,
                    "output_tokens": self.estado.output_tokens,
                },
            }


# Singleton acessivel pelo pipeline (analogo a METRICS).
BUDGET = BudgetController()
