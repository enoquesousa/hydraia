"""Pre-flight checks: valida ambiente ANTES de iniciar o pipeline.

Falha rapido (`PreflightError`) em vez de mascarar problemas com "analise parcial".
Cada check retorna uma `CheckResult` estruturada — facil de logar/serializar.

Checks executados:
  * chaves de API (Gemini, OpenAI, OpenWeather) — formato basico + nao-placeholder
  * LM Studio — `/v1/models` responde com pelo menos um modelo de chat
  * artefatos do HydroLSTM — `hydro_lstm_melhor.pt` + `hydro_lstm_meta.json` + scalers
  * arquivos de contexto — `docs/Crises_Hidricas_SP.md`

Modos:
  STRICT    — toda falha e fatal (default em producao)
  PERMISSIVO — falhas viram warnings (fallback para modo simulado / dry-run)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import requests

from src.utils.env_loader import API_KEY_PLACEHOLDERS, EnvLoader
from src.utils.http_security import (
    exigir_resposta_sem_redirecionamento,
    validar_url_servico,
)
from src.utils.paths import HydraPaths

logger = logging.getLogger(__name__)


# ─── Tipos ───────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    nome: str
    ok: bool
    mensagem: str = ""
    obrigatorio: bool = True
    detalhes: Dict = field(default_factory=dict)


class PreflightError(RuntimeError):
    """Levantado em modo STRICT quando ao menos um check obrigatorio falha."""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _api_key_valida(valor: Optional[str], chave: str) -> bool:
    """Aceita uma chave so se ela: existir, nao for placeholder, ter >=10 chars."""
    if not valor:
        return False
    v = valor.strip()
    if len(v) < 10:
        return False
    return (
        v.lower() not in API_KEY_PLACEHOLDERS
        and v.lower() != f"sua_chave_{chave.lower()}_aqui"
    )


def _check_chave(env_var: str, *, obrigatoria: bool = True) -> CheckResult:
    valor = EnvLoader.get(env_var)
    ok = _api_key_valida(valor, env_var)
    msg = "configurada" if ok else "ausente ou placeholder"
    return CheckResult(
        nome=f"env:{env_var}",
        ok=ok,
        mensagem=msg,
        obrigatorio=obrigatoria,
    )


def _check_lm_studio(url: Optional[str] = None, timeout_s: float = 5.0) -> CheckResult:
    import os
    try:
        base = validar_url_servico(
            url or os.environ.get("LM_STUDIO_URL") or "http://localhost:1234/v1",
            permitir_http_loopback=True,
        )
    except ValueError:
        return CheckResult("lm_studio", False, "configuracao de URL invalida")
    try:
        resp = requests.get(f"{base}/models", timeout=timeout_s, allow_redirects=False)
        exigir_resposta_sem_redirecionamento(resp)
        resp.raise_for_status()
        modelos = resp.json().get("data", [])
        chat = [m for m in modelos if "embed" not in m.get("id", "").lower()]
        if not chat:
            return CheckResult(
                nome="lm_studio",
                ok=False,
                mensagem="LM Studio acessivel mas sem modelo de chat carregado",
                obrigatorio=True,
                detalhes={"url": base, "n_modelos": len(modelos)},
            )
        return CheckResult(
            nome="lm_studio",
            ok=True,
            mensagem=f"online with {len(chat)} modelo(s) de chat",
            obrigatorio=True,
            detalhes={"url": base, "modelos": [m.get("id") for m in chat]},
        )
    except (requests.exceptions.RequestException, ValueError, RuntimeError, TypeError, AttributeError):
        return CheckResult(
            nome="lm_studio",
            ok=False,
            mensagem="inacessivel ou resposta HTTP invalida",
            obrigatorio=True,
            detalhes={"url": base},
        )


def _check_modelo_lstm() -> CheckResult:
    pt = HydraPaths.TIME_SERIES / "hydro_lstm_melhor.pt"
    meta = HydraPaths.TIME_SERIES / "hydro_lstm_meta.json"
    sx = HydraPaths.DATASETS / "scaler_X.pkl"
    sy = HydraPaths.DATASETS / "scaler_y.pkl"
    faltando = [str(p.name) for p in (pt, meta, sx, sy) if not p.exists()]
    if faltando:
        return CheckResult(
            nome="hydro_lstm",
            ok=False,
            mensagem=f"artefatos ausentes: {', '.join(faltando)}",
            obrigatorio=True,
            detalhes={"faltando": faltando},
        )
    return CheckResult(
        nome="hydro_lstm",
        ok=True,
        mensagem="modelo treinado e scalers presentes",
        obrigatorio=True,
    )


def _check_contexto_md() -> CheckResult:
    md = HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md"
    if not md.exists():
        return CheckResult(
            nome="contexto_md",
            ok=False,
            mensagem=f"arquivo ausente: {md}",
            obrigatorio=True,
        )
    return CheckResult(
        nome="contexto_md",
        ok=True,
        mensagem=f"{md.stat().st_size // 1024} KB",
        obrigatorio=True,
    )


def _check_sabesp_ping(timeout_s: float = 8.0) -> CheckResult:
    """Tenta pingar a API SABESP (nao bloqueia o pipeline em modo offline)."""
    url = "https://mananciais.sabesp.com.br/api/v4/dados/ultima-data"
    try:
        resp = requests.get(url, timeout=timeout_s, allow_redirects=False)
        exigir_resposta_sem_redirecionamento(resp)
        resp.raise_for_status()
        data = resp.json()
        if data.get("succeeded"):
            return CheckResult(
                nome="sabesp_api",
                ok=True,
                mensagem=f"online (ultima data: {data.get('data')})",
                obrigatorio=False,
                detalhes={"ultima_data": data.get("data")},
            )
        return CheckResult(
            nome="sabesp_api",
            ok=False,
            mensagem="API respondeu sem 'succeeded=true'",
            obrigatorio=False,
        )
    except (requests.exceptions.RequestException, ValueError, RuntimeError, TypeError, AttributeError):
        return CheckResult(
            nome="sabesp_api",
            ok=False,
            mensagem="offline ou resposta HTTP invalida",
            obrigatorio=False,
        )


def _check_openweather_ping(api_key: Optional[str], timeout_s: float = 5.0) -> CheckResult:
    """Tenta pingar a API OpenWeather com uma chamada minimal (geocoding)."""
    if not api_key:
        return CheckResult("openweather_api", False, "sem chave OpenWeather", obrigatorio=False)
    # Sao Paulo (Praca da Se)
    url = "https://api.openweathermap.org/geo/1.0/reverse"
    params = {"lat": -23.5505, "lon": -46.6333, "limit": 1, "appid": api_key}
    try:
        resp = requests.get(url, params=params, timeout=timeout_s, allow_redirects=False)
        exigir_resposta_sem_redirecionamento(resp)
        if resp.status_code == 401:
            return CheckResult("openweather_api", False, "chave invalida (401)", obrigatorio=True)
        resp.raise_for_status()
        return CheckResult("openweather_api", True, "online e chave valida", obrigatorio=False)
    except (requests.exceptions.RequestException, RuntimeError):
        return CheckResult(
            "openweather_api", False, "erro de conexao ou resposta HTTP recusada", obrigatorio=False,
        )


def _check_gemini_modelo(
    api_key: Optional[str],
    modelo: str,
    fallbacks: Sequence[str] = (),
) -> CheckResult:
    """Sondagem barata: lista modelos disponiveis e verifica se o nome bate (e fallbacks)."""
    if not api_key:
        return CheckResult("gemini_modelo", False, "sem chave Gemini", obrigatorio=True)
    try:
        from google import genai
    except ImportError:
        return CheckResult("gemini_modelo", False, "lib google-genai ausente", obrigatorio=False)
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        ids = {m.name for m in models if hasattr(m, "name")}

        def _match(m_id: str) -> bool:
            return any(m.endswith(m_id) or m == m_id for m in ids)

        if _match(modelo):
            return CheckResult(
                "gemini_modelo", True, f"modelo '{modelo}' acessivel", obrigatorio=True,
                detalhes={"modelo": modelo, "n_disponiveis": len(ids)},
            )

        # Tenta fallbacks
        for fb in fallbacks:
            if _match(fb):
                return CheckResult(
                    "gemini_modelo", True,
                    f"principal '{modelo}' indisponivel — fallback '{fb}' OK",
                    obrigatorio=True,
                    detalhes={"modelo_principal": modelo, "fallback_em_uso": True, "fallback": fb},
                )

        return CheckResult(
            "gemini_modelo", False,
            f"modelo '{modelo}' nao acessivel; {len(fallbacks)} fallbacks indisponiveis",
            obrigatorio=True,
            detalhes={"modelo": modelo, "amostra_disponiveis": sorted(list(ids))[:5]},
        )
    except Exception as e:
        return CheckResult("gemini_modelo", False, f"falha de sondagem: {type(e).__name__}",
                           obrigatorio=True)


def _check_openai_modelo(api_key: Optional[str], modelo: str, fallbacks: List[str]) -> CheckResult:
    """Sondagem: tenta listar modelos OpenAI; se principal indisponivel, verifica
    fallbacks. Nao consome tokens de chat — apenas /models.
    """
    if not api_key:
        return CheckResult("openai_modelo", False, "sem chave OpenAI", obrigatorio=True)
    try:
        from openai import OpenAI
    except ImportError:
        return CheckResult("openai_modelo", False, "lib openai ausente", obrigatorio=False)
    try:
        client = OpenAI(api_key=api_key)
        ids = {m.id for m in client.models.list().data}
        if modelo in ids:
            return CheckResult(
                "openai_modelo", True, f"modelo '{modelo}' acessivel", obrigatorio=True,
                detalhes={"modelo": modelo, "fallback_em_uso": False},
            )
        # Tenta fallbacks
        for fb in fallbacks:
            if fb in ids:
                return CheckResult(
                    "openai_modelo", True,
                    f"principal '{modelo}' indisponivel — fallback '{fb}' OK",
                    obrigatorio=True,
                    detalhes={"modelo_principal": modelo, "fallback_em_uso": True, "fallback": fb},
                )
        return CheckResult(
            "openai_modelo", False,
            f"nenhum modelo acessivel (principal '{modelo}' + {len(fallbacks)} fallbacks)",
            obrigatorio=True,
            detalhes={"modelo": modelo, "fallbacks_tentados": fallbacks},
        )
    except Exception as e:
        return CheckResult("openai_modelo", False, f"falha de sondagem: {type(e).__name__}",
                           obrigatorio=True)


# ─── Pipeline pre-flight ─────────────────────────────────────────────────────

def executar_preflight(
    *,
    exigir_lm_studio: bool = True,
    exigir_lstm: bool = True,
    exigir_sabesp_online: bool = False,
    exigir_openweather: bool = False,
    sondar_modelos_llm: bool = False,
) -> List[CheckResult]:
    """Executa todos os checks e retorna a lista de resultados.

    Args:
        exigir_lm_studio/lstm/sabesp_online/openweather: marcam o check como
            obrigatorio (falha em strict). False = informativo.
        sondar_modelos_llm: se True, faz uma chamada-teste minimal nos SDKs
            Gemini/OpenAI para verificar acesso ao modelo escolhido. Custa uma
            requisicao a /models — sem tokens de chat. Habilitado em `run`
            mas desligado por default em `validate` para nao depender de rede.
    """
    EnvLoader()
    resultados: List[CheckResult] = []

    # Chaves de API
    resultados.append(_check_chave("GEMINI_ULTRA_API_KEY", obrigatoria=True))
    resultados.append(_check_chave("OPENAI_GPT5_API_KEY", obrigatoria=True))
    resultados.append(_check_chave("OPENWEATHER_API_KEY", obrigatoria=exigir_openweather))

    # OpenWeather Ping (opcional por default)
    if exigir_openweather or EnvLoader.get("OPENWEATHER_API_KEY"):
        r = _check_openweather_ping(EnvLoader.get("OPENWEATHER_API_KEY"))
        r.obrigatorio = exigir_openweather
        resultados.append(r)

    # LM Studio
    r = _check_lm_studio()
    r.obrigatorio = exigir_lm_studio
    resultados.append(r)

    # Modelo LSTM
    r = _check_modelo_lstm()
    r.obrigatorio = exigir_lstm
    resultados.append(r)

    # Contexto cientifico
    resultados.append(_check_contexto_md())

    # SABESP (informativo por padrao)
    r = _check_sabesp_ping()
    r.obrigatorio = exigir_sabesp_online
    resultados.append(r)

    # Sondagem de modelos LLM (lista /models — sem gastar tokens)
    if sondar_modelos_llm:
        try:
            from src.core.config import carregar_config
            cfg = carregar_config()

            g = cfg.get("gemini_ultra", {})
            gemini_model = g.get("model", "gemini-2.5-pro")
            gemini_fb = g.get("fallback_models", [])

            o = cfg.get("openai_gpt5", {})
            gpt_model = o.get("model", "gpt-4o")
            gpt_fb = o.get("fallback_models", [])

            resultados.append(_check_gemini_modelo(
                EnvLoader.get("GEMINI_ULTRA_API_KEY"), gemini_model, gemini_fb
            ))
            resultados.append(_check_openai_modelo(
                EnvLoader.get("OPENAI_GPT5_API_KEY"), gpt_model, gpt_fb
            ))
        except Exception:
            resultados.append(CheckResult(
                "modelos_llm_sondagem", False, "erro ao sondar modelos", obrigatorio=False,
            ))

    return resultados


def imprimir_resumo(resultados: List[CheckResult]) -> None:
    """Imprime um resumo legivel dos checks no stdout."""
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT — Validacao do ambiente HydraIA")
    print("=" * 60)
    for r in resultados:
        icone = "OK   " if r.ok else ("ERRO " if r.obrigatorio else "AVISO")
        flag = "[obrig.]" if r.obrigatorio else "[opcio.]"
        print(f"  [{icone}] {flag} {r.nome:<28} {r.mensagem}")
    print("=" * 60 + "\n")


def validar_ou_falhar(
    resultados: List[CheckResult],
    *,
    strict: bool = True,
) -> None:
    """Em modo strict, levanta `PreflightError` se algum check obrigatorio falhou."""
    falhas_obrig = [r for r in resultados if r.obrigatorio and not r.ok]
    if not falhas_obrig:
        return
    if strict:
        nomes = ", ".join(r.nome for r in falhas_obrig)
        raise PreflightError(
            f"Pre-flight falhou em {len(falhas_obrig)} check(s) obrigatorio(s): {nomes}"
        )
    logger.warning("Pre-flight passou em modo permissivo, mas %d check(s) obrigatorio(s) falharam.",
                   len(falhas_obrig))


def relatorio_dict(resultados: List[CheckResult]) -> Dict:
    """Versao serializavel (para logs estruturados / endpoints HTTP)."""
    return {
        "ok": all(r.ok or not r.obrigatorio for r in resultados),
        "checks": [asdict(r) for r in resultados],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    rs = executar_preflight()
    imprimir_resumo(rs)
    try:
        validar_ou_falhar(rs, strict=True)
        print("✅ Ambiente OK para execucao.")
    except PreflightError as e:
        print(f"❌ {e}")
        raise SystemExit(1)
