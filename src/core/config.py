"""Carga e validacao de configuracao do HydraIA.

Centraliza tudo que envolve `config_apis.json` + variaveis de ambiente:
  * `carregar_config()` — le e valida o JSON; aplica overrides do .env.
  * `selecionar_modelo()` — tenta o modelo principal e, em caso de falha
    de autenticacao/permissao, percorre a lista de fallbacks.
  * Schema basico do config — nao permite valores fora dos tipos esperados.

A motivacao: o pipeline antigo carregava JSON cru e usava qualquer modelo
configurado. Se a conta nao tivesse acesso, o pipeline morria sem alerta
claro. Agora ha valida.de boot e fallback automatico.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils.env_loader import EnvLoader
from src.utils.paths import HydraPaths

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    pass


# ─── Schema basico ───────────────────────────────────────────────────────────

_SCHEMA_REQUIRED_TOP = {"openweather", "gemini_ultra", "openai_gpt5"}

_DEFAULT_FALLBACKS = {
    "openai_gpt5": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "gemini_ultra": ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
}


def _validar_dict_secao(secao: str, d: Any, campos: Dict[str, type]) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"secao '{secao}' deveria ser dict, e {type(d).__name__}")
    for k, t in campos.items():
        if k in d and not isinstance(d[k], t):
            raise ConfigError(
                f"{secao}.{k} deveria ser {t.__name__}, e {type(d[k]).__name__}"
            )


def carregar_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Le e valida `config_apis.json`. Levanta ConfigError em problemas.

    Aplica overrides do .env:
      * GEMINI_MODEL  → gemini_ultra.model
      * GPT_MODEL     → openai_gpt5.model
    """
    EnvLoader()
    p = path or (HydraPaths.BASE / "config_apis.json")
    if not p.exists():
        raise ConfigError(f"config nao encontrada: {p}")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"config_apis.json invalido: {e}") from e

    falt = _SCHEMA_REQUIRED_TOP - set(cfg.keys())
    if falt:
        raise ConfigError(f"secoes obrigatorias ausentes: {sorted(falt)}")

    _validar_dict_secao("openweather", cfg["openweather"], {
        "base_url": str, "timeout": int, "retry_attempts": int,
    })
    _validar_dict_secao("gemini_ultra", cfg["gemini_ultra"], {
        "model": str, "max_tokens": int, "temperature": (int, float),
    })
    _validar_dict_secao("openai_gpt5", cfg["openai_gpt5"], {
        "model": str, "max_tokens": int, "temperature": (int, float),
    })

    # Garantir lista de fallbacks (cria se nao existir)
    for chave, defaults in _DEFAULT_FALLBACKS.items():
        if chave in cfg:
            fallbacks = cfg[chave].get("fallback_models")
            if fallbacks is None:
                cfg[chave]["fallback_models"] = list(defaults)
            elif not isinstance(fallbacks, list):
                raise ConfigError(f"{chave}.fallback_models deveria ser lista")

    # Overrides do .env
    gemini_env = EnvLoader.get("GEMINI_MODEL")
    gpt_env = EnvLoader.get("GPT_MODEL")
    if gemini_env:
        cfg["gemini_ultra"]["model"] = gemini_env
    if gpt_env:
        cfg["openai_gpt5"]["model"] = gpt_env

    return cfg


# ─── Selecao de modelo com fallback ──────────────────────────────────────────

def selecionar_modelo(
    cfg_secao: Dict,
    *,
    sondador: Callable[[str], bool],
) -> Optional[str]:
    """Tenta o modelo principal e, se nao disponivel, percorre fallbacks.

    `sondador(model_id)` deve retornar True se o modelo esta acessivel pela
    conta. Funcao mantida abstrata para facilitar testes (sem rede).
    """
    candidatos: List[str] = []
    principal = cfg_secao.get("model")
    if principal:
        candidatos.append(principal)
    candidatos.extend(cfg_secao.get("fallback_models", []))
    candidatos = list(dict.fromkeys(candidatos))  # de-dup mantendo ordem

    for m in candidatos:
        try:
            if sondador(m):
                if m != principal:
                    logger.warning("[config] Modelo principal '%s' indisponivel — usando fallback '%s'",
                                   principal, m)
                return m
        except Exception as e:
            logger.warning("[config] Sondagem de '%s' falhou: %s", m, e)
    return None
