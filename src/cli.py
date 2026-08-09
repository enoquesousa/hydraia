"""HydraIA CLI — pontos de entrada unificados.

Subcomandos:
  validate   Roda preflight do ambiente (chaves, LM Studio, LSTM, contexto).
  dry-run    Monta os prompts (hidro/meteo/sintese) e simula sem chamar LLMs.
  run        Executa o pipeline.
                  --offline                    usa snapshot SABESP se API cair
                  --modo-simulado              dados sinteticos (demo offline)
                  --skip-llm                   gera relatorio sem chamar LLMs
                  --economico                  aplica BUDGET economico
                  --sem-baseline-fallback      aborta se LSTM faltar
                  --sondar-modelos             pre-flight valida modelos via SDK
                  --skip-preflight             pula o preflight (uso interno)
                  --horizonte-dias N           horizonte de previsao (default 7)
                  --json                       loga em JSON line
  metrics    Imprime as metricas in-process da ultima execucao.
  preflight  Sinonimo de validate, mas com codigo de saida nao-zero em falhas.

Uso:
    python -m src.cli validate --offline
    python -m src.cli dry-run --offline
    python -m src.cli run                         # pipeline real
    python -m src.cli run --modo-simulado --skip-llm  # demo TCC offline
    python -m src.cli run --offline --economico       # economico, snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from src.utils.tls import configurar_certificados_nativos

# Aplicacao executavel: use as CAs confiadas pelo sistema antes de instanciar
# clientes requests/httpx dos provedores externos.
configurar_certificados_nativos()

from src.utils.budget import BUDGET
from src.utils.observabilidade import (
    METRICS,
    novo_correlation_id,
    setup_logging,
    versao_codigo,
)
from src.utils.paths import HydraPaths

logger = logging.getLogger(__name__)


# ─── Sub: validate / preflight ───────────────────────────────────────────────

def cmd_validate(args: argparse.Namespace) -> int:
    from src.core.preflight import (
        executar_preflight,
        imprimir_resumo,
        relatorio_dict,
        validar_ou_falhar,
    )

    rs = executar_preflight(
        exigir_lm_studio=not args.offline,
        exigir_lstm=not args.offline,
        exigir_sabesp_online=False,
        exigir_openweather=False,
        sondar_modelos_llm=getattr(args, "sondar_modelos", False),
    )
    imprimir_resumo(rs)
    if args.json:
        print(json.dumps(relatorio_dict(rs), ensure_ascii=False, indent=2))
    if args.strict:
        try:
            validar_ou_falhar(rs, strict=True)
        except Exception as e:
            print(f"❌ {e}")
            return 1
    return 0 if all(r.ok or not r.obrigatorio for r in rs) else 1


# ─── Sub: dry-run ────────────────────────────────────────────────────────────

def cmd_dry_run(args: argparse.Namespace) -> int:
    """Monta os prompts e exibe — nao chama nenhum LLM. Util para revisao no TCC."""
    from src.core.orquestrador import (
        _prompt_hidrologico,
        _prompt_meteorologico,
        _prompt_sintese,
    )
    from src.core.integracao import (
        carregar_contexto_markdown,
        coletar_dados_openweather,
        coletar_dados_sabesp_reais,
        gerar_previsoes_ml,
    )

    novo_correlation_id(prefix="dryrun")
    print("=" * 60)
    print("  HydraIA — DRY RUN (sem chamadas LLM)")
    print("=" * 60)

    if args.offline:
        sabesp = coletar_dados_sabesp_reais(modo_simulado=True)
        weather = coletar_dados_openweather(modo_simulado=True)
        contexto = "(contexto cientifico mockado para dry-run offline)"
    else:
        sabesp = coletar_dados_sabesp_reais(allow_snapshot=True)
        weather = coletar_dados_openweather()
        md = HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md"
        contexto = carregar_contexto_markdown(md) if md.exists() else "(contexto nao encontrado)"

    cantareira = sabesp["sistemas"].get("cantareira", {})
    resumo = sabesp.get("resumo", {})
    estado = {
        "data_referencia": cantareira.get("data", date.today().isoformat()),
        "volume_cantareira_pct": cantareira.get("volume_percentual", 0),
        "volume_cantareira_hm3": cantareira.get("volume_hm3"),
        "chuva_mm": cantareira.get("chuva_mm"),
        "vazao_natural_m3s": cantareira.get("vazao_natural_m3s"),
        "vazao_jusante_m3s": cantareira.get("vazao_jusante_m3s"),
        "situacao_operacional": "DryRun",
        "macrometropole": {
            "total_sistemas": resumo.get("total_sistemas"),
            "volume_medio_pct": resumo.get("volume_medio_percentual"),
            "sistemas_abaixo_50pct": resumo.get("sistemas_abaixo_50pct"),
            "sistemas_criticos": resumo.get("sistemas_criticos"),
        },
        "modo_dados": sabesp.get("modo", "real"),
    }

    previsoes = gerar_previsoes_ml(
        horizonte=7,
        permitir_baseline_fallback=True,
        modo_simulado_dados=args.offline,
    )

    p_hidro = _prompt_hidrologico(estado, previsoes)
    p_meteo = _prompt_meteorologico(estado, previsoes, weather)
    convergencia_mock = {"status": "convergencia", "risco_consolidado": 3,
                         "confianca_ensemble_pct": 75, "risco_hidro": 3, "risco_meteo": 3,
                         "delta": 0}
    p_sintese = _prompt_sintese(estado, previsoes, weather, contexto, None, None, convergencia_mock)

    print("\n--- PROMPT HIDRO ---")
    print(p_hidro)
    print("\n--- PROMPT METEO ---")
    print(p_meteo)
    print("\n--- PROMPT SINTESE (resumido) ---")
    print(p_sintese[:1000] + ("..." if len(p_sintese) > 1000 else ""))
    print(f"\nTamanho prompt sintese: {len(p_sintese)} chars")
    return 0


# ─── Sub: run ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    # Se o usuario passou um CID customizado, usamos ele.
    if getattr(args, "cid", None):
        from src.utils.observabilidade import _CORRELATION_ID
        _CORRELATION_ID.set(args.cid)
    else:
        novo_correlation_id(prefix="run")

    METRICS.reset()
    BUDGET.reset()

    if args.economico:
        BUDGET.aplicar_modo_economico()
        logger.info("[run] modo economico ativado: %s", BUDGET.snapshot()["limites"])

    from src.core.integracao import executar_pipeline_integrado
    from src.core.preflight import (
        executar_preflight,
        imprimir_resumo,
        validar_ou_falhar,
    )

    skip_llm = args.dry_run
    allow_snapshot = args.allow_snapshot

    if not args.skip_preflight:
        # Exigimos LM Studio so se vamos chamar LLM (sintetizador local).
        # Exigimos LSTM so quando nao temos fallback (--sem-baseline-fallback)
        # E nao estamos em modo simulado.
        exigir_lm = (not skip_llm) and (not args.modo_simulado)
        exigir_lstm = (not args.modo_simulado) and args.sem_baseline_fallback
        exigir_sabesp = not (allow_snapshot or args.modo_simulado)
        rs = executar_preflight(
            exigir_lm_studio=exigir_lm,
            exigir_lstm=exigir_lstm,
            exigir_sabesp_online=exigir_sabesp,
            sondar_modelos_llm=args.sondar_modelos and not skip_llm,
        )
        imprimir_resumo(rs)
        try:
            validar_ou_falhar(rs, strict=True)
        except Exception as e:
            print(f"❌ Preflight falhou: {e}")
            return 2

    md = HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md"
    print(f"[run] Versao codigo: {versao_codigo()}")
    relatorio = executar_pipeline_integrado(
        caminho_md=md,
        horizonte_dias=args.horizonte_dias,
        modo_simulado_dados=args.modo_simulado,
        allow_snapshot=allow_snapshot,
        skip_llm=skip_llm,
        permitir_baseline_fallback=not args.sem_baseline_fallback,
    )
    print("\n" + "=" * 60)
    print(f"RELATORIO HYDRAIA")
    print("=" * 60)
    print(relatorio)
    print("\n--- METRICAS ---")
    print(json.dumps(METRICS.snapshot(), indent=2, ensure_ascii=False))
    print("\n--- ORCAMENTO ---")
    print(json.dumps(BUDGET.snapshot(), indent=2, ensure_ascii=False))
    return 0


# ─── Sub: metrics ────────────────────────────────────────────────────────────

def cmd_metrics(args: argparse.Namespace) -> int:
    print(json.dumps({
        "metrics": METRICS.snapshot(),
        "budget": BUDGET.snapshot(),
    }, indent=2, ensure_ascii=False))
    return 0


# ─── Parser ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hydraia", description="HydraIA — CLI unificado")
    p.add_argument("--log-json", "--json-logs", action="store_true", dest="json_logs", help="logging em JSON line")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="executa preflight do ambiente")
    pv.add_argument("--strict", action="store_true", help="codigo de saida 1 em falha")
    pv.add_argument("--json", action="store_true", help="imprime resumo JSON")
    pv.add_argument("--offline", "--allow-snapshot", action="store_true", help="nao exige LM Studio/LSTM")
    pv.add_argument("--sondar-modelos", action="store_true",
                    help="testa acesso aos modelos Gemini/OpenAI via SDK (gasta 1 listagem)")
    pv.set_defaults(func=cmd_validate)

    pp = sub.add_parser("preflight", help="alias de validate --strict")
    pp.add_argument("--json", action="store_true")
    pp.add_argument("--offline", "--allow-snapshot", action="store_true")
    pp.add_argument("--sondar-modelos", action="store_true")
    pp.set_defaults(func=lambda a: cmd_validate(argparse.Namespace(
        strict=True, json=a.json, offline=getattr(a, "offline", False) or getattr(a, "allow_snapshot", False),
        sondar_modelos=a.sondar_modelos,
    )))

    pd = sub.add_parser("dry-run", help="monta prompts sem chamar LLMs")
    pd.add_argument("--offline", "--allow-snapshot", action="store_true", dest="allow_snapshot", help="usa dados simulados")
    pd.set_defaults(func=cmd_dry_run)

    pr = sub.add_parser("run", help="executa o pipeline completo")
    pr.add_argument("--offline", "--allow-snapshot", action="store_true", dest="allow_snapshot",
                    help="usa snapshot SABESP se API cair")
    pr.add_argument("--modo-simulado", action="store_true",
                    help="dados SABESP/OpenWeather sinteticos (demo offline)")
    pr.add_argument("--skip-llm", "--dry-run", action="store_true", dest="dry_run",
                    help="gera relatorio skeleton sem chamar LLMs")
    pr.add_argument("--economico", action="store_true",
                    help="aplica BUDGET economico (max 6 chamadas LLM, 5min, 10k tokens)")
    pr.add_argument("--sem-baseline-fallback", action="store_true",
                    help="aborta se HydroLSTM nao estiver disponivel")
    pr.add_argument("--sondar-modelos", action="store_true",
                    help="preflight valida modelos LLM via SDK")
    pr.add_argument("--skip-preflight", action="store_true",
                    help="pula o preflight (uso interno)")
    pr.add_argument("--horizonte-dias", type=int, default=7,
                    help="dias de previsao (default 7)")
    pr.add_argument("--cid", type=str, help="Correlation ID customizado")
    pr.set_defaults(func=cmd_run)

    pm = sub.add_parser("metrics", help="imprime metricas in-process")
    pm.set_defaults(func=cmd_metrics)

    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    # Garantir que as flags de log funcionem corretamente
    json_logs = getattr(args, "log_json", False) or getattr(args, "json_logs", False)
    setup_logging(json_logs=json_logs)

    if args.cmd != "metrics":
        # cmd_run vai sobrescrever se tiver --cid, mas inicializamos aqui para o boot
        novo_correlation_id(prefix=args.cmd or "cli")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
