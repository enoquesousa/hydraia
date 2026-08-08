"""Integração HydraIA: dados SABESP reais + HydroLSTM + Ensemble LLM.

Pipeline:

1. Coletar dados REAIS da API SABESP (reservatórios de SP)
2. Gerar previsão de v_util (%) via HydroLSTM treinado nos 26 anos de histórico
   (fallback: simulação se o modelo ainda não estiver treinado)
3. Carregar contexto científico do TCC a partir de um arquivo .md
4. Coletar previsão meteorológica do OpenWeather (enriquecimento)
5. Ensemble: Gemini (hidrológico) + GPT (meteorológico) em paralelo → validação determinística → LLM local (síntese)
6. Receber relatório técnico estruturado sobre risco hídrico

Para treinar o modelo:
    python preparar_dataset.py
    python treinar_modelo_ts.py
"""

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.sabesp import SabespCollector, SISTEMAS_SABESP
from src.api.openweather import OpenWeatherCollector, coletar_simulado as coletar_weather_simulado
from src.core.orquestrador import OrquestradorLLM
from src.data.estatisticas_prob import (
    anomalias_volume_atual,
    previsao_empirica_meses,
    probabilidade_dia_referencia,
)
from src.utils.cache import FileCache, validador_sabesp, validador_openweather
from src.utils.env_loader import EnvLoader
from src.utils.paths import HydraPaths

BASE_DIR = HydraPaths.BASE
logger = logging.getLogger(__name__)

_CACHE_DIR = BASE_DIR / "storage" / "raw" / "cache"
_CACHE_TTL = 3600  # 1 hora em segundos
_cache = FileCache(_CACHE_DIR, ttl_s=_CACHE_TTL)
_cache_sabesp = FileCache(_CACHE_DIR, ttl_s=_CACHE_TTL, validator=validador_sabesp)
_cache_weather = FileCache(_CACHE_DIR, ttl_s=_CACHE_TTL, validator=validador_openweather)


# Aliases legados (testes em test_cache.py importam por estes nomes).
def _ler_cache(nome: str) -> Optional[Dict]:
    return _cache.get(nome)


def _salvar_cache(nome: str, dados: Dict) -> None:
    _cache.set(nome, dados)


def carregar_contexto_markdown(caminho_md: Path) -> str:
    """Lê o arquivo .md com a pesquisa do TCC e retorna o texto bruto.

    Resiliente: se o arquivo não existir, retorna um aviso em vez de travar o pipeline.
    """
    if not caminho_md.exists():
        logger.warning("Arquivo de contexto nao encontrado em %s. Usando placeholder.", caminho_md)
        return "(Aviso: contexto científico não carregado — arquivo docs/Crises_Hidricas_SP.md ausente)"
    return caminho_md.read_text(encoding="utf-8")


def coletar_dados_sabesp_reais(
    *,
    allow_snapshot: bool = False,
    modo_simulado: bool = False,
) -> Dict[str, Any]:
    """Coleta dados dos reservatorios da SABESP (com cache de 1h validado).

    Args:
        allow_snapshot: se True, em caso de falha da API tenta o ultimo
            snapshot persistido em vez de propagar erro.
        modo_simulado: se True, ignora API + cache e retorna dados sinteticos
            (uso: dry-run / demos offline).
    """
    if modo_simulado:
        collector = SabespCollector(auto_save=False)
        return collector.coletar_simulado()

    cache = _cache_sabesp.get("sabesp")
    if cache:
        return cache

    logger.info("Coletando dados da API SABESP...")
    collector = SabespCollector(auto_save=True)
    dados = collector.coletar_macrometropole_completa(allow_snapshot=allow_snapshot)

    if not dados or not dados.get("sistemas"):
        raise RuntimeError("Dados SABESP indisponiveis. Operacao abortada para preservar integridade.")

    logger.info("Dados coletados: %d sistemas (modo=%s)",
                len(dados["sistemas"]), dados.get("modo", "real"))
    for sistema, info in dados.get("sistemas", {}).items():
        if info:
            nome = SISTEMAS_SABESP.get(sistema, sistema)
            vol = info.get("volume_percentual", "N/A")
            logger.info("  %s: %s%%", nome, vol)

    _cache_sabesp.set("sabesp", dados)
    return dados


def coletar_dados_openweather(
    lat: float = -23.5505,
    lon: float = -46.6333,
    dias: int = 7,
    *,
    modo_simulado: bool = False,
) -> Optional[Dict]:
    """Coleta previsao meteorologica (cache de 1h validado).

    Retorna None se API estiver indisponivel — pipeline segue, e enriquecimento.
    Em `modo_simulado`, retorna dados sinteticos.
    """
    if modo_simulado:
        return coletar_weather_simulado(dias=dias)

    cache = _cache_weather.get("openweather")
    if cache:
        return cache

    logger.info("Coletando previsao meteorologica (OpenWeather)...")
    EnvLoader()
    try:
        config_path = BASE_DIR / "config_apis.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        coletor = OpenWeatherCollector(config["openweather"])
        dados = coletor.coletar_previsao(lat, lon, dias=dias)
        if dados and dados.get("previsoes"):
            logger.info("OpenWeather: %d dias de previsao coletados", dados['total_previsoes'])
            current = dados.get("current") or {}
            if current:
                logger.info("OpenWeather agora: %s°C, chuva 1h: %smm",
                            current.get('temp'), current.get('rain_1h', 0.0))
            _cache_weather.set("openweather", dados)
        return dados
    except Exception as e:
        logger.warning("OpenWeather indisponivel (nao bloqueia o pipeline): %s", e)
        return None


def gerar_previsoes_ml(
    horizonte: int = 7,
    *,
    permitir_baseline_fallback: bool = True,
    modo_simulado_dados: bool = False,
    dados_sabesp: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """Gera previsoes do HydroLSTM. Em caso de falha, opcionalmente cai para
    um baseline (persistence/climatologia) em vez de abortar o pipeline.

    Em `modo_simulado_dados=True`, sempre usa o baseline (nao tenta o LSTM).
    """
    if modo_simulado_dados:
        logger.info("[ml] modo_simulado_dados=True — usando baseline persistence")
        return _baseline_previsoes(
            horizonte,
            usar_climatologia=False,
            dados_sabesp=dados_sabesp,
        )

    try:
        from src.data.inferencia_ts import prever
        resultado = prever(horizonte_dias=horizonte)
        if resultado is None:
            raise RuntimeError("Modelo ML nao retornou resultado.")

        cantareira = resultado.get("Cantareira", [])
        previsoes: List[Dict[str, float]] = []
        for i, dia_prev in enumerate(cantareira[:horizonte]):
            row: Dict[str, float] = {
                "dia": float(i + 1),
                "chuva_mm": 0.0,
                "volume_cantareira_pct": dia_prev["v_util_pct"],
                "fonte_previsao": "hydrolstm",
            }
            # Intervalos e probabilidades (se disponíveis)
            if "p10" in dia_prev:
                row["volume_cantareira_p10"] = dia_prev["p10"]
            if "p90" in dia_prev:
                row["volume_cantareira_p90"] = dia_prev["p90"]
            if "prob_sub_30_pct" in dia_prev:
                row["prob_cantareira_sub_30_pct"] = dia_prev["prob_sub_30_pct"]
            if "prob_sub_50_pct" in dia_prev:
                row["prob_cantareira_sub_50_pct"] = dia_prev["prob_sub_50_pct"]
            if "prob_acima_80_pct" in dia_prev:
                row["prob_cantareira_acima_80_pct"] = dia_prev["prob_acima_80_pct"]
            for sistema, dias in resultado.items():
                if sistema != "Cantareira" and i < len(dias):
                    chave = f"volume_{sistema.lower().replace(' ', '_')}_pct"
                    row[chave] = dias[i]["v_util_pct"]
                    if "p10" in dias[i]:
                        row[f"{chave}_p10"] = dias[i]["p10"]
                    if "p90" in dias[i]:
                        row[f"{chave}_p90"] = dias[i]["p90"]
                    if "prob_sub_30_pct" in dias[i]:
                        row[f"{chave}_prob_sub_30_pct"] = dias[i]["prob_sub_30_pct"]
                    if "prob_sub_50_pct" in dias[i]:
                        row[f"{chave}_prob_sub_50_pct"] = dias[i]["prob_sub_50_pct"]
                    if "prob_acima_80_pct" in dias[i]:
                        row[f"{chave}_prob_acima_80_pct"] = dias[i]["prob_acima_80_pct"]
            previsoes.append(row)
        return previsoes

    except Exception as e:
        if permitir_baseline_fallback:
            logger.warning("[ml] HydroLSTM indisponivel (%s) — usando baseline local", e)
            return _baseline_previsoes(
                horizonte,
                usar_climatologia=True,
                dados_sabesp=dados_sabesp,
            )
        logger.error("[ml] Erro na inferencia ML: %s", e)
        raise


def _baseline_previsoes(
    horizonte: int,
    *,
    usar_climatologia: bool = False,
    dados_sabesp: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """Gera previsoes simples (climatologia ou persistence) a partir dos dados SABESP
    atuais. Usado como fallback quando o LSTM nao esta disponivel.
    """
    import numpy as np

    # A demo simulada precisa ser deterministica e portavel. A climatologia so
    # e consultada quando o chamador a solicita explicitamente (fallback real).
    media_mes = None
    if usar_climatologia:
        try:
            ds_path = HydraPaths.DATASETS / "sabesp_dataset.npz"
            if ds_path.exists():
                with np.load(ds_path) as data:
                    if "media_mes" in data:
                        media_mes = data["media_mes"].copy()  # (12, n_targets)
                        logger.info("[ml] Baseline climatologico (media_mes) carregado.")
        except (OSError, ValueError, KeyError) as e:
            logger.warning("[ml] Nao foi possivel carregar media_mes: %s", e)

    dados_referencia = dados_sabesp if dados_sabesp is not None else _cache_sabesp.get("sabesp")
    sistemas = (dados_referencia or {}).get("sistemas") or {}
    mes_atual = date.today().month

    def _vol(chave: str, target_idx: int, default: float) -> float:
        # Prioridade 1: media_mes para o mes atual
        if media_mes is not None and target_idx < media_mes.shape[1]:
            return float(media_mes[mes_atual - 1, target_idx])

        # Prioridade 2: Ultimo valor real do cache (persistence)
        s = sistemas.get(chave)
        if s and isinstance(s.get("volume_percentual"), (int, float)):
            return float(s["volume_percentual"])
        return default

    # Mapeamento de indices (baseado na ordem do preparar_dataset.py)
    # Cantareira=0, Alto Tiete=1, Guarapiranga=2, Cotia=3, Rio Grande=4, Rio Claro=5
    cantareira = _vol("cantareira", 0, 50.0)
    alto_tiete = _vol("alto_tiete", 1, 60.0)
    guarapiranga = _vol("guarapiranga", 2, 75.0)
    cotia = _vol("cotia", 3, 45.0)
    rio_grande = _vol("rio_grande", 4, 70.0)
    rio_claro = _vol("rio_claro", 5, 65.0)

    previsoes: List[Dict[str, float]] = []
    fonte = "climatologia" if media_mes is not None else "persistence"

    for i in range(horizonte):
        previsoes.append({
            "dia": float(i + 1),
            "chuva_mm": 0.0,
            "volume_cantareira_pct": cantareira,
            "volume_alto_tiete_pct": alto_tiete,
            "volume_guarapiranga_pct": guarapiranga,
            "volume_cotia_pct": cotia,
            "volume_rio_grande_pct": rio_grande,
            "volume_rio_claro_pct": rio_claro,
            "fonte_previsao": f"baseline_{fonte}",
        })
    return previsoes


def _resumo_prob_previsoes(previsoes: List[Dict[str, float]]) -> Dict[str, float]:
    """Extrai um resumo probabilístico das previsões do Cantareira (dia 7 e pior caso)."""
    if not previsoes:
        return {}

    campos_prob = {
        "prob_cantareira_sub_30_pct": "prob_sub_30_pct",
        "prob_cantareira_sub_50_pct": "prob_sub_50_pct",
        "prob_cantareira_acima_80_pct": "prob_acima_80_pct",
    }
    resumo = {}

    # Usa dia +7 (ou último disponível) como referência operacional
    dia_ref = previsoes[min(6, len(previsoes) - 1)]
    for c, alias in campos_prob.items():
        if c in dia_ref:
            resumo[f"{alias}_h7"] = float(dia_ref[c])

    # Pior caso no horizonte
    for c, alias in campos_prob.items():
        valores = [p.get(c) for p in previsoes if p.get(c) is not None]
        if valores:
            resumo[f"{alias}_max"] = float(max(valores))

    # Intervalos do primeiro dia (curto prazo)
    d0 = previsoes[0]
    for k in ("p10", "p50", "p90"):
        if k in d0:
            resumo[f"{k}_h1"] = d0[k]

    return resumo


def executar_pipeline_integrado(
    caminho_md: Path,
    horizonte_dias: int = 7,
    *,
    modo_simulado_dados: bool = False,
    allow_snapshot: bool = False,
    skip_llm: bool = False,
    permitir_baseline_fallback: bool = True,
    usar_dados_reais: bool = True,   # mantido por compat
) -> str:
    """Executa o pipeline integrado: SABESP → HydroLSTM → ensemble LLM → relatorio.

    Args:
        caminho_md: caminho para o arquivo Markdown com contexto cientifico.
        horizonte_dias: numero de dias para previsao.
        modo_simulado_dados: se True, usa SABESP+OpenWeather sinteticos (demos offline).
        allow_snapshot: se True, em caso de falha da API SABESP usa o ultimo
            snapshot persistido em vez de abortar.
        skip_llm: se True, gera apenas um esqueleto de relatorio sem chamar LLMs
            (uso: dry-run de demonstracao).
        permitir_baseline_fallback: se True, caso o HydroLSTM nao esteja
            disponivel usa baselines (persistence/climatologia) em vez de abortar.
        usar_dados_reais: legado — equivale a `not modo_simulado_dados`.
    """
    if not usar_dados_reais:
        modo_simulado_dados = True

    from src.utils.observabilidade import METRICS, novo_correlation_id, versao_codigo
    if METRICS.snapshot()["counters"] == {} and METRICS.snapshot()["histograms"] == {}:
        # primeiro uso desta execucao: garante CID inicializado
        novo_correlation_id(prefix="run")

    logger.info("[pipeline] versao_codigo=%s modo=%s allow_snapshot=%s skip_llm=%s",
                versao_codigo(),
                "simulado" if modo_simulado_dados else "real",
                allow_snapshot, skip_llm)

    t_start = time.perf_counter()

    with METRICS.timer("pipeline.contexto"):
        contexto = carregar_contexto_markdown(caminho_md)
    logger.info("[pipeline] contexto carregado (%.2fs)", time.perf_counter() - t_start)
    t_checkpoint = time.perf_counter()

    with METRICS.timer("pipeline.sabesp"):
        dados_sabesp = coletar_dados_sabesp_reais(
            allow_snapshot=allow_snapshot,
            modo_simulado=modo_simulado_dados,
        )
    logger.info("[pipeline] dados SABESP coletados (modo=%s, %.2fs)",
                dados_sabesp.get("modo"), time.perf_counter() - t_checkpoint)
    t_checkpoint = time.perf_counter()

    with METRICS.timer("pipeline.previsoes_ml"):
        previsoes = gerar_previsoes_ml(
            horizonte=horizonte_dias,
            permitir_baseline_fallback=permitir_baseline_fallback,
            modo_simulado_dados=modo_simulado_dados,
            dados_sabesp=dados_sabesp,
        )
    logger.info("[pipeline] previsoes ML geradas: %d dias (%.2fs)",
                len(previsoes), time.perf_counter() - t_checkpoint)
    t_checkpoint = time.perf_counter()

    with METRICS.timer("pipeline.openweather"):
        dados_weather = coletar_dados_openweather(
            dias=horizonte_dias,
            modo_simulado=modo_simulado_dados,
        )
    logger.info("[pipeline] dados OpenWeather coletados (%.2fs)", time.perf_counter() - t_checkpoint)
    t_checkpoint = time.perf_counter()

    cantareira = dados_sabesp["sistemas"].get("cantareira")
    if not cantareira:
        raise ValueError("❌ Erro: dados do Sistema Cantareira ausentes na coleta.")

    volume_pct = cantareira.get("volume_percentual", 0)
    if volume_pct >= 60:
        situacao = "Faixa Normal"
    elif volume_pct >= 40:
        situacao = "Faixa de Atencao"
    elif volume_pct >= 20:
        situacao = "Faixa de Alerta"
    else:
        situacao = "Faixa de Restricao (Critica)"

    modo_dados = dados_sabesp.get("modo", "real")
    fonte = {
        "real": "API SABESP (dados reais)",
        "snapshot": "SABESP (snapshot — fallback offline)",
        "simulado": "SABESP (simulado — demo offline)",
    }.get(modo_dados, f"SABESP ({modo_dados})")

    estado_atual = {
        "data_referencia": cantareira.get("data", date.today().isoformat()),
        "volume_cantareira_pct": volume_pct,
        "volume_cantareira_hm3": cantareira.get("volume_hm3"),
        "chuva_mm": cantareira.get("chuva_mm"),
        "vazao_natural_m3s": cantareira.get("vazao_natural_m3s"),
        "vazao_jusante_m3s": cantareira.get("vazao_jusante_m3s"),
        "situacao_operacional": situacao,
        "fonte_dados": fonte,
        "modo_dados": modo_dados,
        "timestamp_coleta": cantareira.get("timestamp_coleta"),
        "sistemas": dados_sabesp.get("sistemas") or {},
    }

    if dados_sabesp.get("resumo"):
        resumo = dados_sabesp["resumo"]
        estado_atual["macrometropole"] = {
            "total_sistemas": resumo.get("total_sistemas") or resumo.get("total_sistemas_coletados"),
            "volume_medio_pct": resumo.get("volume_medio_percentual"),
            "sistemas_abaixo_50pct": resumo.get("sistemas_abaixo_50pct"),
            "sistemas_criticos": resumo.get("sistemas_criticos") or resumo.get("sistemas_criticos_abaixo_30pct"),
        }

    # Uma demo sintetica nao deve depender de artefatos historicos locais nem
    # misturar dados simulados com climatologia real.
    if not modo_simulado_dados:
        try:
            estado_atual["anomalias_historicas"] = anomalias_volume_atual(dados_sabesp)
        except Exception as e:
            logger.warning("Falha ao calcular anomalias historicas: %s", e)

        # Probabilidades empíricas (dia de referência e meses futuros)
        data_ref_str = estado_atual["data_referencia"]
        try:
            estado_atual["prob_hist_dia"] = probabilidade_dia_referencia(data_ref_str)
            estado_atual["prob_hist_futuro"] = previsao_empirica_meses(
                data_ref_str,
                offsets_dias=(1, 30, 90),
            )
        except Exception as e:
            logger.warning("Falha ao calcular probabilidades historicas: %s", e)

    # Flag para indicar se HydroLSTM está ativo (sem fallback baseline)
    fontes_previsao = {p.get("fonte_previsao") for p in previsoes}
    estado_atual["modelo_ml_disponivel"] = fontes_previsao == {"hydrolstm"}
    estado_atual["fonte_previsao"] = next(iter(fontes_previsao), "desconhecida")

    if skip_llm:
        return _gerar_relatorio_skeleton(estado_atual, previsoes, dados_weather, modo_dados)

    with METRICS.timer("pipeline.orquestrador"):
        orquestrador = OrquestradorLLM.from_config()
        # Resumo probabilístico das previsões (se disponíveis)
        prob_resumo = _resumo_prob_previsoes(previsoes)
        if prob_resumo:
            estado_atual["probabilidades_previsao"] = prob_resumo
        resposta = orquestrador.gerar_relatorio(
            estado_atual=estado_atual,
            previsoes_ml=previsoes,
            weather=dados_weather,
            contexto_cientifico=contexto,
        )
    return resposta


def _gerar_relatorio_skeleton(
    estado: Dict, previsoes: list, weather: Optional[Dict], modo_dados: str,
) -> str:
    """Skeleton textual para `--skip-llm` / dry-run sem orquestrador."""
    linhas = [
        f"# RELATORIO HYDRAIA — DRY-RUN ({modo_dados})",
        f"Data referencia: {estado.get('data_referencia')}",
        f"Volume Cantareira: {estado.get('volume_cantareira_pct')}% — {estado.get('situacao_operacional')}",
        f"Fonte: {estado.get('fonte_dados')}",
        "",
        f"## Previsao de volume (fonte: {estado.get('fonte_previsao', 'desconhecida')})",
    ]
    for p in previsoes:
        linhas.append(f"  Dia +{int(p.get('dia', 0))}: {p.get('volume_cantareira_pct'):.1f}%")
    linhas.append("")
    if weather and weather.get("previsoes"):
        linhas.append("## Previsao Meteorologica (resumo)")
        for prev in weather["previsoes"][:7]:
            linhas.append(f"  {prev.get('data_hora', '')[:10]}: chuva={prev.get('precipitacao_mm', 0):.1f}mm")
    linhas.append("")
    linhas.append("(LLMs nao foram chamados — modo --skip-llm)")
    return "\n".join(linhas)


def main() -> None:
    """Funcao principal: roda o pipeline e imprime o relatorio gerado."""
    from src.utils.observabilidade import (
        METRICS, novo_correlation_id, setup_logging,
    )
    setup_logging()
    novo_correlation_id(prefix="main")
    METRICS.reset()

    print("=" * 60)
    print("HydraIA - Sistema de Previsao de Crises Hidricas")
    print("=" * 60)

    caminho_md = BASE_DIR / "docs" / "Crises_Hidricas_SP.md"

    relatorio = executar_pipeline_integrado(
        caminho_md=caminho_md,
        horizonte_dias=7,
        modo_simulado_dados=False,
        allow_snapshot=True,
        skip_llm=False,
    )

    print("\n" + "=" * 60)
    print("RELATORIO GERADO PELO ENSEMBLE LLM (Gemini + GPT + Mistral)")
    print("=" * 60 + "\n")
    print(relatorio)


if __name__ == "__main__":
    main()
