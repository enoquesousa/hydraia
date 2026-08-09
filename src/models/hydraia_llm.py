"""Cliente HTTP para LLM local via LM Studio (API OpenAI-compatible).

Este cliente é **model-agnostic**: descobre dinamicamente qual modelo está
carregado no servidor LM Studio (Qwen3.5, Mistral, Llama, etc.) e usa
ele automaticamente. Isso desacopla o pipeline de uma escolha rígida de modelo
e permite trocar pelo painel do LM Studio sem alterar código.

Configuração via .env:
  LM_STUDIO_URL=http://localhost:1234/v1     (opcional, padrão localhost:1234)
  LM_STUDIO_MODEL=auto                       (auto = descobre dinamicamente; ou nome explícito)
  LM_STUDIO_TIMEOUT=300                      (timeout da síntese em segundos)
  LM_STUDIO_REASONING=off                    (preserva o orçamento para o relatório final)

Uso:
    from src.models.hydraia_llm import ask_local_llm
    resposta = ask_local_llm("Qual a capital da França?", max_tokens=100)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT_S = 300  # síntese pode demorar com modelos parcialmente descarregados na GPU

_modelo_cache: Optional[str] = None


def _base_url() -> str:
    return os.environ.get("LM_STUDIO_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout_s() -> float:
    try:
        return max(1.0, float(os.environ.get("LM_STUDIO_TIMEOUT", DEFAULT_TIMEOUT_S)))
    except ValueError:
        logger.warning("LM_STUDIO_TIMEOUT invalido; usando %ss", DEFAULT_TIMEOUT_S)
        return float(DEFAULT_TIMEOUT_S)


def _native_chat_url() -> str:
    override = os.environ.get("LM_STUDIO_NATIVE_CHAT_URL")
    if override:
        return override.rstrip("/")
    base = _base_url()
    root = base[:-3] if base.endswith("/v1") else base
    return f"{root}/api/v1/chat"


def _reasoning_mode() -> str:
    modo = os.environ.get("LM_STUDIO_REASONING", "off").strip().lower()
    permitidos = {"off", "low", "medium", "high", "on"}
    if modo not in permitidos:
        logger.warning("LM_STUDIO_REASONING invalido; usando off")
        return "off"
    return modo


def _descobrir_modelo() -> str:
    """Consulta /v1/models e retorna o id do primeiro modelo carregado.

    Cache em memória — uma única chamada de descoberta por processo.
    """
    global _modelo_cache
    if _modelo_cache is not None:
        return _modelo_cache

    forcado = os.environ.get("LM_STUDIO_MODEL")
    if forcado and forcado.lower() != "auto":
        _modelo_cache = forcado
        return forcado

    try:
        resp = requests.get(f"{_base_url()}/models", timeout=5)
        resp.raise_for_status()
        modelos = resp.json().get("data", [])
        # Filtrar embeddings — usamos apenas chat models
        chat = [m for m in modelos if "embed" not in m.get("id", "").lower()]
        if not chat:
            raise RuntimeError("Nenhum modelo de chat carregado no LM Studio.")
        _modelo_cache = chat[0]["id"]
        logger.info("[LM Studio] Modelo detectado: %s", _modelo_cache)
        return _modelo_cache
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(_msg_lm_studio_offline()) from e


def _msg_lm_studio_offline() -> str:
    return (
        "Não foi possível conectar ao LM Studio.\n"
        "Verifique que:\n"
        "  1. O LM Studio está aberto\n"
        "  2. O servidor local está ativo (Developer → Local Server → Start)\n"
        "  3. Há um modelo de chat carregado\n"
        f"  4. A URL {_base_url()} está acessível"
    )


def ask_local_llm(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> str:
    """Envia um prompt para o LLM local. Detecta o modelo carregado se não for fornecido.

    Args:
        prompt: Texto do prompt.
        max_tokens: Limite de tokens na resposta.
        temperature: Aleatoriedade da geração (0.0–1.0).
        model: Nome explícito; se None, descobre dinamicamente.
    """
    modelo = model or _descobrir_modelo()
    native_payload = {
        "model": modelo,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "reasoning": _reasoning_mode(),
        "store": False,
    }

    try:
        response = requests.post(
            _native_chat_url(),
            json=native_payload,
            timeout=_timeout_s(),
        )
        # Modelos sem controle de raciocinio podem rejeitar esse campo.
        if response.status_code == 400:
            native_payload.pop("reasoning", None)
            response = requests.post(
                _native_chat_url(),
                json=native_payload,
                timeout=_timeout_s(),
            )
        # Compatibilidade com versoes antigas do LM Studio sem /api/v1/chat.
        if response.status_code == 404:
            legacy_payload = {
                "model": modelo,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            response = requests.post(
                f"{_base_url()}/chat/completions",
                json=legacy_payload,
                timeout=_timeout_s(),
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        response.raise_for_status()
        data = response.json()
        usados = int((data.get("stats") or {}).get("total_output_tokens") or 0)
        if usados >= max_tokens:
            raise RuntimeError(
                f"LM Studio atingiu o limite de {max_tokens} tokens; "
                "a resposta foi descartada para evitar relatorio truncado."
            )
        mensagens = [
            item.get("content", "").strip()
            for item in data.get("output", [])
            if item.get("type") == "message" and item.get("content")
        ]
        if not mensagens:
            raise RuntimeError("LM Studio concluiu a geracao sem mensagem final.")
        return "\n\n".join(mensagens)

    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(_msg_lm_studio_offline()) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"Timeout ({_timeout_s():g}s) ao gerar resposta. "
            f"Modelo {modelo} pode ser muito grande para esse prompt — "
            f"reduza max_tokens ou aumente LM_STUDIO_TIMEOUT."
        ) from e


# Alias retrocompatível: o código histórico chama `ask_mistral` mas o cliente
# agora é genérico. Mantemos o nome para não quebrar `orquestrador.py`.
ask_mistral = ask_local_llm
