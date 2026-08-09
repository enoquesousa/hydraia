"""LLM Specialists — Strategy + Template Method para chamada de provedores.

Cada provedor de LLM (Gemini, OpenAI GPT, ...) é uma estratégia intercambiável.
A classe abstrata `LLMSpecialist` encapsula o esqueleto comum (validação de chave,
retry com backoff, parsing JSON, validação de schema, timeout, circuit breaker
e watchdog de tamanho de resposta). As subclasses preenchem apenas `_invocar`,
isolando o detalhe de cada SDK.

Princípios aplicados:
  - Strategy Pattern: cada provedor é uma estratégia plugável.
  - Template Method: `analisar()` define o esqueleto; `_invocar()` é o gancho.
  - Open/Closed: adicionar Anthropic Claude amanhã = criar `ClaudeSpecialist`,
    sem alterar o `OrquestradorLLM`.
  - Single Responsibility: cada classe sabe só falar com seu provedor.
  - Fail Fast: timeout local, circuit breaker e validacao de chave evitam
    pendurar o pipeline e gastar tokens em providers caídos.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Optional

from src.utils.budget import BUDGET, BudgetExceeded
from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)


# Constantes de seguranca para watchdog/limites
DEFAULT_TIMEOUT_S = 45.0          # tempo maximo por chamada ao provedor
DEFAULT_MAX_RESPONSE_CHARS = 64_000  # corta respostas absurdas (proxy de tokens)
DEFAULT_MAX_PROMPT_CHARS = 60_000    # ~15k tokens, limite saudavel para a maioria dos modelos


# ─── Helpers de parsing/validação ────────────────────────────────────────────

def extrair_json(texto: str) -> Optional[Dict]:
    """Extrai o primeiro bloco JSON válido de uma resposta de LLM."""
    if not texto:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL)
    candidato = match.group(1) if match else None
    if candidato is None:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        candidato = match.group(0) if match else texto
    try:
        return json.loads(candidato)
    except (json.JSONDecodeError, TypeError):
        return None


def validar_analise(resultado: Optional[Dict], nome: str) -> Optional[Dict]:
    """Valida e normaliza o JSON retornado por um LLM especialista.

    - `nivel_risco` deve ser inteiro entre 1 e 5
    - `confianca_pct` é clampado para [0, 100]; ausente → 0
    Retorna None para JSON inválido ou fora dos intervalos.
    """
    if not resultado or not isinstance(resultado, dict):
        logger.warning("[%s] JSON invalido ou ausente — descartando analise.", nome)
        return None

    risco = resultado.get("nivel_risco")
    try:
        risco = int(risco)
    except (TypeError, ValueError):
        logger.warning("[%s] nivel_risco invalido (%r) — descartando analise.", nome, risco)
        return None

    if not 1 <= risco <= 5:
        logger.warning("[%s] nivel_risco fora do intervalo 1-5 (%d) — descartando.", nome, risco)
        return None

    resultado["nivel_risco"] = risco

    conf = resultado.get("confianca_pct", 0)
    try:
        resultado["confianca_pct"] = max(0, min(100, int(conf)))
    except (TypeError, ValueError):
        resultado["confianca_pct"] = 0

    return resultado


def truncar_prompt(prompt: str, max_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> str:
    """Trunca o prompt no inicio (preserva as instrucoes finais que orientam o JSON).

    Mantem os primeiros 1/3 e os ultimos 2/3 do limite, com marcador no meio.
    O final do prompt costuma conter as instrucoes de formato — nao podemos perder.
    """
    if len(prompt) <= max_chars:
        return prompt
    head = max_chars // 3
    tail = max_chars - head - 64
    marcador = "\n\n[...conteudo truncado por watchdog...]\n\n"
    return prompt[:head] + marcador + prompt[-tail:]


# ─── Hierarquia de Specialists ───────────────────────────────────────────────


class StopRetry(Exception):
    """Sinal: este erro é permanente, não tente novamente."""


class LLMSpecialist(ABC):
    """Abstração de um especialista LLM.

    Encapsula: chave de API, modelo, hiperparâmetros, retry, timeout local,
    circuit breaker por instancia e watchdog de tamanho de resposta.
    Subclasses implementam apenas `_invocar` (chamada bruta ao SDK do provedor).
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 2,
        backoff_s: float = 3.0,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
        breaker_threshold: int = 3,
        breaker_recovery_s: float = 60.0,
        nome: Optional[str] = None,
        strict_schema: bool = False,
        campos_obrigatorios: Optional[set] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self.max_response_chars = max_response_chars
        self.max_prompt_chars = max_prompt_chars
        self.nome = nome or self.__class__.__name__
        self.last_model_used: Optional[str] = None
        self.strict_schema = strict_schema
        self.campos_obrigatorios = set(campos_obrigatorios or [])
        self.breaker = CircuitBreaker(
            nome=self.nome,
            failure_threshold=breaker_threshold,
            recovery_s=breaker_recovery_s,
        )

    # Template Method
    def analisar(self, prompt: str) -> Optional[Dict]:
        """Pipeline: validar key → checar budget → truncar → invocar → parse → validar."""
        if not self.api_key:
            logger.warning("[%s] Chave de API ausente — pulando.", self.nome)
            return None

        prompt_seguro = truncar_prompt(prompt, self.max_prompt_chars)
        if len(prompt_seguro) < len(prompt):
            logger.warning(
                "[%s] Prompt truncado de %d para %d chars (limite %d).",
                self.nome, len(prompt), len(prompt_seguro), self.max_prompt_chars,
            )

        # Checagem proativa de orcamento — se exceder, nao chama o provider.
        ok, motivo = BUDGET.can_call(prompt=prompt_seguro)
        if not ok:
            logger.error("[%s] Orcamento bloqueia chamada: %s", self.nome, motivo)
            return None

        texto = self._invocar_com_retry(prompt_seguro)
        if not texto:
            # Mesmo em falha, contabilizamos a tentativa para nao "vazar" budget.
            try:
                BUDGET.account(prompt=prompt_seguro, resposta="", strict=False)
            except BudgetExceeded:
                pass
            return None

        if len(texto) > self.max_response_chars:
            logger.warning(
                "[%s] Resposta truncada por watchdog (%d → %d chars).",
                self.nome, len(texto), self.max_response_chars,
            )
            texto = texto[: self.max_response_chars]

        try:
            BUDGET.account(prompt=prompt_seguro, resposta=texto, strict=True)
        except BudgetExceeded as e:
            logger.error("[%s] Orcamento excedido apos resposta: %s", self.nome, e)
            # Retornamos a analise mesmo assim — ja gastamos a chamada;
            # mas marcamos o budget como excedido para que chamadas futuras parem.

        analise = validar_analise(extrair_json(texto), self.nome)

        # Modo strict: descartar resposta com campos obrigatorios faltando
        if self.strict_schema and analise is not None and self.campos_obrigatorios:
            falt = self.campos_obrigatorios - set(analise.keys())
            if falt:
                logger.warning(
                    "[%s] strict_schema: campos obrigatorios faltando %s — descartando.",
                    self.nome, sorted(falt),
                )
                return None
        return analise

    def _invocar_com_retry(self, prompt: str) -> Optional[str]:
        """Retry linear com backoff. Erros transitórios são tolerados; erros
        permanentes (StopRetry) ou breaker aberto curto-circuitam imediatamente."""
        for tentativa in range(1, self.max_retries + 2):
            try:
                return self._chamar_com_timeout(prompt)
            except CircuitOpenError as e:
                logger.error("[%s] %s — abortando.", self.nome, e)
                return None
            except StopRetry as e:
                logger.error("[%s] Erro permanente: %s", self.nome, e)
                return None
            except Exception as e:
                if tentativa <= self.max_retries:
                    logger.warning("[%s] Tentativa %d falhou (%s) — aguardando %.0fs...",
                                   self.nome, tentativa, e, self.backoff_s)
                    time.sleep(self.backoff_s)
                else:
                    logger.error("[%s] Todas as %d tentativas falharam: %s",
                                 self.nome, self.max_retries + 1, e)
        return None

    def _chamar_com_timeout(self, prompt: str) -> str:
        """Aplica timeout local + circuit breaker em torno de `_invocar`.

        Usa ThreadPoolExecutor com 1 worker para nao depender de signal (que nao
        funciona em threads no Windows). O thread fica dangling em caso de
        timeout, mas o pipeline segue — preferivel a travar o processo.

        Em Python 3.11+, `concurrent.futures.TimeoutError is builtins.TimeoutError`,
        entao distinguimos timeout real (future ainda nao concluiu) de uma
        TimeoutError levantada dentro do `_invocar` via `fut.done()`.
        """
        def _alvo() -> str:
            return self.breaker.call(self._invocar, prompt)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"llm-{self.nome}") as ex:
            fut = ex.submit(_alvo)
            try:
                return fut.result(timeout=self.timeout_s)
            except FuturesTimeoutError as e:
                if not fut.done():
                    # timeout real do `fut.result(...)`: marca falha extra no
                    # breaker porque o `_invocar` ainda esta rodando e nao
                    # vai chegar a `breaker.call` para registrar.
                    self.breaker._record_failure()
                    raise TimeoutError(
                        f"[{self.nome}] timeout de {self.timeout_s:.0f}s excedido"
                    ) from e
                # `_invocar` ja terminou com TimeoutError — breaker.call ja
                # registrou; apenas propaga sem duplo registro.
                raise

    @abstractmethod
    def _invocar(self, prompt: str) -> str:
        """Chamada bruta ao SDK do provedor. Lance `StopRetry` para erros que
        não devem ser retentados (ex: chave inválida, modelo inexistente)."""


# ─── Implementações concretas ────────────────────────────────────────────────

class GeminiSpecialist(LLMSpecialist):
    """Especialista Google Gemini (lib `google-genai`)."""

    def _invocar(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        self.last_model_used = self.model
        return response.text or ""


class GPTSpecialist(LLMSpecialist):
    """Especialista OpenAI GPT (lib `openai`).

    Modelos gpt-5* e o-series não aceitam `temperature` — detectamos pelo
    prefixo do nome do modelo e omitimos o parâmetro proativamente.
    """

    @staticmethod
    def _aceita_temperature(model: str) -> bool:
        nome = model.lower()
        return not (nome.startswith("gpt-5") or nome.startswith("o1") or nome.startswith("o3"))

    def _invocar(self, prompt: str) -> str:
        from openai import OpenAI, AuthenticationError, BadRequestError, NotFoundError

        client = OpenAI(api_key=self.api_key)

        # Lista de modelos a tentar: principal + fallbacks (se existirem)
        candidatos = [self.model]
        for m in getattr(self, "_fallback_models", []):
            if m not in candidatos:
                candidatos.append(m)

        ultimo_erro: Optional[Exception] = None
        for modelo in candidatos:
            payload: Dict = {
                "model": modelo,
                "input": prompt,
                "max_output_tokens": self.max_tokens,
            }
            if modelo.lower().startswith("gpt-5"):
                payload["reasoning"] = {"effort": "none"}
            if self._aceita_temperature(modelo):
                payload["temperature"] = self.temperature

            try:
                response = client.responses.create(**payload)
                if modelo != self.model:
                    logger.warning("[%s] Usando fallback '%s' (principal '%s' indisponivel)",
                                   self.nome, modelo, self.model)
                self.last_model_used = modelo
                return response.output_text or ""
            except BadRequestError as e:
                msg = str(e).lower()
                if "temperature" in msg and "temperature" in payload:
                    logger.warning("[%s] Modelo %s rejeitou 'temperature' — retentando sem.",
                                   self.nome, modelo)
                    payload.pop("temperature", None)
                    try:
                        response = client.responses.create(**payload)
                        self.last_model_used = modelo
                        return response.output_text or ""
                    except Exception as e2:
                        ultimo_erro = e2
                        continue
                if "model" in msg and ("does not exist" in msg or "not found" in msg or "access" in msg):
                    logger.warning("[%s] Modelo '%s' indisponivel — tentando proximo fallback", self.nome, modelo)
                    ultimo_erro = e
                    continue
                raise StopRetry(f"BadRequest nao relacionado a temperature: {e}") from e
            except NotFoundError as e:
                logger.warning("[%s] Modelo '%s' nao encontrado — tentando fallback", self.nome, modelo)
                ultimo_erro = e
                continue
            except AuthenticationError as e:
                raise StopRetry(f"Chave OpenAI invalida: {e}") from e

        # Se chegou aqui, nenhum modelo funcionou
        raise StopRetry(f"Nenhum modelo OpenAI acessivel (tentados: {candidatos}). Ultimo erro: {ultimo_erro}")
