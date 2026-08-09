"""
Orquestrador LLM — HydraIA
===========================
Ensemble de três especialistas para análise hídrica de precisão cirúrgica:

  1. Especialista hidrológico  (Gemini 2.5 Pro, por padrão) → SABESP + HydroLSTM
  2. Especialista meteorológico (GPT, por padrão)            → OpenWeather × reservatórios
  3. Cross-validation                                       → mede convergência/divergência
  4. Sintetizador local         (LLM via LM Studio)         → relatório final privado

Quanto menor o delta entre os riscos estimados, maior a confiança do relatório.
Quando os modelos divergem, o sintetizador recebe a divergência explicitamente
e a usa como sinal de incerteza epistêmica no relatório.

Padrões aplicados:
  - Strategy Pattern (especialistas plugáveis via `LLMSpecialist`)
  - Dependency Injection (orquestrador recebe especialistas prontos, não os cria)
  - Open/Closed Principle (adicionar Anthropic Claude = nova subclasse, zero
    alteração nesta classe)
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from src.core.llm_specialists import (
    GeminiSpecialist,
    GPTSpecialist,
    LLMSpecialist,
    extrair_json,        # re-exportado para compatibilidade com testes legados
    validar_analise,     # re-exportado
)
from src.core.validacao_semantica import (
    avaliar_analise,
    convergencia_semantica,
)
from src.utils.env_loader import EnvLoader
from src.models.hydraia_llm import ask_mistral
from src.utils.observabilidade import METRICS, correlation_id, hash_prompt, versao_codigo
from src.utils.paths import HydraPaths

logger = logging.getLogger(__name__)

_SISTEMAS_LSTM = (
    ("volume_cantareira_pct", "Cantareira"),
    ("volume_alto_tiete_pct", "Alto Tietê"),
    ("volume_guarapiranga_pct", "Guarapiranga"),
    ("volume_cotia_pct", "Cotia"),
    ("volume_rio_grande_pct", "Rio Grande"),
    ("volume_rio_claro_pct", "Rio Claro"),
)

_NOMES_CANONICOS = {
    "cantareira": "Cantareira",
    "alto_tiete": "Alto Tietê",
    "guarapiranga": "Guarapiranga",
    "cotia": "Cotia",
    "rio_grande": "Rio Grande",
    "rio_claro": "Rio Claro",
    "sao_lourenco": "São Lourenço",
}


# Aliases para manter compatibilidade com testes existentes que importam por _nome.
_extrair_json = extrair_json
_validar_analise = validar_analise


# ─── Formatadores de prompt ──────────────────────────────────────────────────

def _formatar_sabesp(estado: Dict) -> str:
    m = estado.get("macrometropole", {})
    sistemas = estado.get("sistemas", {})

    # Métricas agregadas
    vols = [s.get("volume_percentual") for s in sistemas.values() if s.get("volume_percentual") is not None]
    if vols:
        import numpy as np
        agg = (
            f"MÉTRICAS AGREGADAS:\n"
            f"  Média: {np.mean(vols):.2f}% | Desvio-Padrão: {np.std(vols):.2f}%\n"
            f"  Mín: {np.min(vols):.2f}% | Máx: {np.max(vols):.2f}%\n"
        )
    else:
        agg = ""

    # Detalhe por sistema
    detalhes = []
    for slug, info in sistemas.items():
        detalhes.append(
            f"  - {info.get('sistema', slug)}: {info.get('volume_percentual'):.2f}% "
            f"(Chuva: {info.get('chuva_mm', 0)}mm, Vazão Nat: {info.get('vazao_natural_m3s', 0)}m³/s)"
        )
    detalhes_txt = "\n".join(detalhes)

    return (
        f"Data: {estado.get('data_referencia')}\n"
        f"{agg}\n"
        f"SITUAÇÃO POR SISTEMA:\n{detalhes_txt}\n\n"
        f"RESUMO MACROMETRÓPOLE:\n"
        f"Volume médio: {m.get('volume_medio_pct')}% | Sistemas críticos (<30%): {m.get('sistemas_criticos', 0)}"
    )


def _formatar_lstm(previsoes: list) -> str:
    sistemas = {
        "volume_cantareira_pct": "Cantareira",
        "volume_alto_tiete_pct": "Alto Tietê",
        "volume_guarapiranga_pct": "Guarapiranga",
        "volume_cotia_pct": "Cotia",
        "volume_rio_grande_pct": "Rio Grande",
        "volume_rio_claro_pct": "Rio Claro",
    }
    linhas = []
    for p in previsoes:
        dia = int(p.get("dia", 0))
        for campo, nome in sistemas.items():
            if campo not in p:
                continue
            prefixo = "volume_cantareira" if nome == "Cantareira" else campo
            vol = p[campo]
            extras = [f"pontual={vol:.1f}%"]
            quantis = [p.get(f"{prefixo}_{q}") for q in ("p10", "p50", "p90")]
            if all(v is not None for v in quantis):
                extras.append(
                    f"p10={quantis[0]:.1f}% | p50={quantis[1]:.1f}% | p90={quantis[2]:.1f}%"
                )
            prob_prefixo = "prob_cantareira" if nome == "Cantareira" else f"{campo}_prob"
            prob30 = p.get(f"{prob_prefixo}_sub_30_pct")
            prob50 = p.get(f"{prob_prefixo}_sub_50_pct")
            prob80 = p.get(f"{prob_prefixo}_acima_80_pct")
            if prob30 is not None:
                extras.append(f"P(<30%)={prob30:.1f}%")
            if prob50 is not None:
                extras.append(f"P(<50%)={prob50:.1f}%")
            if prob80 is not None:
                extras.append(f"P(>80%)={prob80:.1f}%")
            linhas.append(f"  Dia +{dia} | {nome}: " + " | ".join(extras))
    return "\n".join(linhas)


def _formatar_weather(weather: Optional[Dict], max_dias: Optional[int] = None) -> str:
    if not weather or not weather.get("previsoes"):
        return "  (dados meteorológicos não disponíveis)"
    linhas = []
    current = weather.get("current") or {}
    if current:
        linhas.append(
            f"  Agora: {current.get('temp')}°C, umidade {current.get('humidity')}%, "
            f"chuva última hora {current.get('rain_1h', 0.0)}mm"
        )
    previsoes = weather["previsoes"][:max_dias] if max_dias else weather["previsoes"]
    for prev in previsoes:
        data = prev.get("data_hora", "")[:10]
        chuva = prev.get("precipitacao_mm") or 0.0
        temp = prev.get("temperatura_c")
        umid = prev.get("umidade_pct")
        desc = prev.get("descricao", "")
        linhas.append(f"  {data}: {temp}°C, umidade {umid}%, chuva {chuva:.1f}mm — {desc}")
    return "\n".join(linhas)


# ─── Renderizacao deterministica das secoes factuais ────────────────────────

def _numero(valor, casas: int = 2, *, sinal: bool = False, sufixo: str = "") -> str:
    if valor is None:
        return "N/D"
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return "N/D"
    formato = f"{{:{'+' if sinal else ''}.{casas}f}}"
    return f"{formato.format(numero)}{sufixo}"


def _estacao(data_referencia: Optional[str]) -> str:
    try:
        mes = datetime.fromisoformat(str(data_referencia)[:10]).month
    except (TypeError, ValueError):
        mes = datetime.now().month
    if 7 <= mes <= 9:
        return "PERÍODO SECO (JAS)"
    if mes in (12, 1, 2):
        return "PERÍODO CHUVOSO (DJF)"
    return "TRANSIÇÃO"


def _offset_historico(estado: Dict, sistema: str, offset: int) -> Dict:
    offsets = (estado.get("prob_hist_futuro") or {}).get(sistema) or {}
    return offsets.get(offset) or offsets.get(str(offset)) or {}


def _secoes_fatuais(
    estado: Dict,
    previsoes: list,
    weather: Optional[Dict],
) -> str:
    """Renderiza tabelas 1-5 sem delegar números ou cobertura à LLM."""
    data_ref = estado.get("data_referencia")
    macro = estado.get("macrometropole") or {}
    partes = [
        "# RELATÓRIO TÉCNICO FINAL: ANÁLISE DE RISCO HÍDRICO - "
        "MACROMETRÓPOLE DE SÃO PAULO",
        f"**Sistema:** HydraIA | **Data de referência:** {data_ref or 'N/D'}",
        "> **Nota metodológica:** probabilidades de evento e confiança do ensemble são "
        "estimativas empíricas/heurísticas não calibradas. Este relatório é um artefato "
        "de pesquisa e não deve fundamentar, isoladamente, decisões operacionais.",
        "### 1) Situação Atual",
        "| Item | Valor |",
        "|---|---|",
        f"| Data de referência | {data_ref or 'N/D'} |",
        f"| Estação climatológica | {_estacao(data_ref)} |",
        f"| Cantareira (%) | {_numero(estado.get('volume_cantareira_pct'), sufixo='%')} |",
        f"| Macrometrópole média (%) | {_numero(macro.get('volume_medio_pct'), sufixo='%')} |",
        f"| Sistemas críticos (<30%) | {macro.get('sistemas_criticos', 'N/D')} |",
        "",
        "### 2) Situação por Sistema e Deltas",
        "| Sistema | Volume (%) | Delta vs p50 mês | Chuva (mm) | "
        "Vazão natural (m³/s) | Anomalia z-score |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    anomalias = estado.get("anomalias_historicas") or {}
    for slug, info in (estado.get("sistemas") or {}).items():
        nome = _NOMES_CANONICOS.get(
            slug, str(info.get("sistema") or slug).removeprefix("Sistema ")
        )
        volume = info.get("volume_percentual")
        p50_mes = _offset_historico(estado, nome, 1).get("p50")
        delta = None if volume is None or p50_mes is None else float(volume) - float(p50_mes)
        zscore = (anomalias.get(nome) or {}).get("zscore")
        partes.append(
            f"| {nome} | {_numero(volume, sufixo='%')} | "
            f"{_numero(delta, sinal=True, sufixo='%')} | "
            f"{_numero(info.get('chuva_mm'))} | "
            f"{_numero(info.get('vazao_natural_m3s'), 3)} | "
            f"{_numero(zscore, 2, sinal=True)} |"
        )

    partes.extend([
        "",
        "### 3) Previsão HydroLSTM (7 dias, por sistema)",
        "| Dia | Sistema | p10 | p50 | p90 | P(<30%) | P(<50%) | P(>80%) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for previsao in previsoes:
        dia = int(previsao.get("dia", 0))
        for campo, nome in _SISTEMAS_LSTM:
            if campo not in previsao:
                continue
            prefixo_q = "volume_cantareira" if nome == "Cantareira" else campo
            prefixo_p = "prob_cantareira" if nome == "Cantareira" else f"{campo}_prob"
            partes.append(
                f"| +{dia} | {nome} | "
                f"{_numero(previsao.get(f'{prefixo_q}_p10'), 1, sufixo='%')} | "
                f"{_numero(previsao.get(f'{prefixo_q}_p50'), 1, sufixo='%')} | "
                f"{_numero(previsao.get(f'{prefixo_q}_p90'), 1, sufixo='%')} | "
                f"{_numero(previsao.get(f'{prefixo_p}_sub_30_pct'), 1, sufixo='%')} | "
                f"{_numero(previsao.get(f'{prefixo_p}_sub_50_pct'), 1, sufixo='%')} | "
                f"{_numero(previsao.get(f'{prefixo_p}_acima_80_pct'), 1, sufixo='%')} |"
            )

    partes.extend([
        "",
        "### 4) Previsão Meteorológica (7 dias)",
        "| Data | Chuva (mm) | Temp (°C) | Umidade (%) | Descrição |",
        "|---|---:|---:|---:|---|",
    ])
    for previsao in ((weather or {}).get("previsoes") or [])[:len(previsoes)]:
        partes.append(
            f"| {str(previsao.get('data_hora') or '')[:10] or 'N/D'} | "
            f"{_numero(previsao.get('precipitacao_mm'), 1)} | "
            f"{_numero(previsao.get('temperatura_c'), 2)} | "
            f"{_numero(previsao.get('umidade_pct'), 0)} | "
            f"{previsao.get('descricao') or 'N/D'} |"
        )

    partes.extend([
        "",
        "### 5) Horizonte Estendido (empírico h+30 e h+90)",
        "| Sistema | h+30 p10/p50/p90 | h+90 p10/p50/p90 |",
        "|---|---|---|",
    ])
    for _, nome in _SISTEMAS_LSTM:
        horizontes = []
        for offset in (30, 90):
            dados = _offset_historico(estado, nome, offset)
            if not dados:
                horizontes.append("N/D")
            else:
                horizontes.append(
                    "/".join(_numero(dados.get(q), 2) for q in ("p10", "p50", "p90"))
                )
        partes.append(f"| {nome} | {horizontes[0]} | {horizontes[1]} |")

    return "\n".join(partes)


def _montar_relatorio_final(
    narrativa: str,
    estado: Dict,
    previsoes: list,
    weather: Optional[Dict],
    *,
    exigir_secoes: bool,
) -> str:
    inicio = re.search(r"(?m)^#{2,3}\s*6\)", narrativa or "")
    if inicio:
        narrativa = narrativa[inicio.start():].strip()
    elif exigir_secoes:
        raise RuntimeError("Sintetizador local nao retornou a secao 6 do relatorio.")

    if exigir_secoes:
        faltantes = [
            numero
            for numero in (6, 7, 8)
            if not re.search(rf"(?m)^#{{2,3}}\s*{numero}\)", narrativa)
        ]
        if faltantes:
            raise RuntimeError(f"Sintetizador local omitiu secoes obrigatorias: {faltantes}")

    return f"{_secoes_fatuais(estado, previsoes, weather)}\n\n{narrativa.strip()}\n"


# ─── Construtores de prompt ──────────────────────────────────────────────────

def _prompt_hidrologico(estado: Dict, previsoes: list) -> str:
    prob = estado.get("probabilidades_previsao", {})
    prob_txt = ""
    if prob:
        prob_txt = (
            "\n## PROBABILIDADES (Cantareira)\n"
            f"Curto prazo (h+1): p10={prob.get('p10_h1', 'N/A')}% | "
            f"p90={prob.get('p90_h1', 'N/A')}%\n"
            f"Até +7d: P(<30%)={prob.get('prob_sub_30_pct_h7', 'N/A')}% | "
            f"P(<50%)={prob.get('prob_sub_50_pct_h7', 'N/A')}% | "
            f"P(>80%)={prob.get('prob_acima_80_pct_h7', 'N/A')}%"
        )

    return f"""Você é um especialista sênior em hidrologia e gestão de recursos hídricos da Macrometrópole de São Paulo.

## DADOS ATUAIS DOS RESERVATÓRIOS (API SABESP)
{_formatar_sabesp(estado)}

## PROJEÇÃO HIDRO-LSTM (volume do Cantareira — próximos {len(previsoes)} dias)
{_formatar_lstm(previsoes)}
{prob_txt}

## TAREFA
Analise exclusivamente a perspectiva hidrológica: estado dos reservatórios, tendência de volume, risco de crise.

Responda APENAS com JSON válido (sem texto antes ou depois):
{{
  "nivel_risco": <inteiro 1 a 5>,
  "tendencia": "<melhora|estavel|piora>",
  "sistemas_criticos": ["<nome>", ...],
  "analise": "<2-3 frases técnicas concisas sobre risco hidrológico>",
  "confianca_pct": <inteiro 0 a 100>
}}

Escala de risco: 1=mínimo, 2=baixo, 3=atenção, 4=alerta, 5=crítico."""


def _prompt_hidrologico_ext(estado: Dict, previsoes: list) -> str:
    """Versão estendida com probabilidades empíricas (histórico) e sazonalidade."""
    prob_hist_dia = estado.get("prob_hist_dia") or {}
    prob_hist_fut = estado.get("prob_hist_futuro") or {}

    extra_hist_dia = ""
    if prob_hist_dia:
        linhas = []
        for sistema, info in prob_hist_dia.items():
            linhas.append(
                f"{sistema}: p10={info.get('p10')} p50={info.get('p50')} p90={info.get('p90')} (doy={info.get('doy')})"
            )
        extra_hist_dia = "\n## HISTÓRICO — DIA DO ANO\n" + "\n".join(linhas)

    extra_hist_fut = ""
    if prob_hist_fut:
        linhas = ["## HISTÓRICO — MESES FUTUROS (empírico)"]
        for sistema, offs in prob_hist_fut.items():
            for off, info in sorted(offs.items()):
                linhas.append(
                    f"{sistema} h+{off}d (mes {info.get('mes_ref')}): "
                    f"p10={info.get('p10')} p50={info.get('p50')} p90={info.get('p90')}"
                )
        extra_hist_fut = "\n" + "\n".join(linhas)

    return _prompt_hidrologico(estado, previsoes) + extra_hist_dia + extra_hist_fut


def _prompt_meteorologico(estado: Dict, previsoes: list, weather: Optional[Dict]) -> str:
    prob_hist_dia = estado.get("prob_hist_dia") or {}
    prob_hist_fut = estado.get("prob_hist_futuro") or {}

    def _fmt_hist() -> str:
        linhas = []
        if prob_hist_dia:
            linhas.append("## HISTÓRICO — DIA DO ANO (hidrológico)")
            for sistema, info in prob_hist_dia.items():
                linhas.append(
                    f"{sistema}: p10={info.get('p10')} p50={info.get('p50')} p90={info.get('p90')} (doy={info.get('doy')})"
                )
        if prob_hist_fut:
            linhas.append("## HISTÓRICO — MESES FUTUROS (empírico hidrológico)")
            for sistema, offs in prob_hist_fut.items():
                for off, info in sorted(offs.items()):
                    linhas.append(
                        f"{sistema} h+{off}d (mes {info.get('mes_ref')}): "
                        f"p10={info.get('p10')} p50={info.get('p50')} p90={info.get('p90')}"
                    )
        if not linhas:
            return ""
        return "\n" + "\n".join(linhas)

    return f"""Você é um especialista em meteorologia e impacto climático em sistemas de abastecimento hídrico.

## DADOS ATUAIS DOS RESERVATÓRIOS (API SABESP)
{_formatar_sabesp(estado)}

## PROJEÇÃO HIDRO-LSTM (volume do Cantareira — próximos {len(previsoes)} dias)
{_formatar_lstm(previsoes)}

## PREVISÃO METEOROLÓGICA (OpenWeather — São Paulo)
{_formatar_weather(weather, max_dias=len(previsoes))}

{_fmt_hist()}

## TAREFA
Avalie como a precipitação prevista impactará os reservatórios nos próximos 7 dias.
Cruze a chuva esperada com a projeção do HydroLSTM.

Responda APENAS com JSON válido (sem texto antes ou depois):
{{
  "nivel_risco": <inteiro 1 a 5>,
  "chuva_total_esperada_mm": <float>,
  "impacto_nos_reservatorios": "<positivo|neutro|negativo>",
  "dias_com_chuva_significativa": <inteiro>,
  "analise": "<2-3 frases técnicas concisas sobre impacto meteorológico>",
  "confianca_pct": <inteiro 0 a 100>
}}

Escala de risco: 1=mínimo, 2=baixo, 3=atenção, 4=alerta, 5=crítico."""


def _prompt_sintese(
    estado: Dict,
    previsoes: list,
    weather: Optional[Dict],
    contexto: str,
    analise_hidro: Optional[Dict],
    analise_meteo: Optional[Dict],
    convergencia: Dict,
) -> str:
    def fmt_analise(a: Optional[Dict], nome: str) -> str:
        if not a:
            return f"  {nome}: indisponível"
        return (
            f"  {nome}:\n"
            f"    Nível de risco: {a.get('nivel_risco')}/5\n"
            f"    Tendência: {a.get('tendencia', a.get('impacto_nos_reservatorios', 'N/A'))}\n"
            f"    Confiança: {a.get('confianca_pct')}%\n"
            f"    Análise: {a.get('analise', '')}"
        )

    conv_status = convergencia.get("status")
    conf_ens = convergencia.get("confianca_ensemble_pct", 0)
    risco_cons = convergencia.get("risco_consolidado", "N/A")
    if conv_status == "convergencia":
        conv_desc = (
            f"CONVERGÊNCIA — ambos os especialistas concordam. "
            f"Risco consolidado: {risco_cons}/5. Confiança ensemble: {conf_ens}%."
        )
    elif conv_status == "divergencia":
        conv_desc = (
            f"DIVERGÊNCIA — risco hidrológico={convergencia.get('risco_hidro')}/5 vs "
            f"meteorológico={convergencia.get('risco_meteo')}/5 (delta={convergencia.get('delta')}). "
            f"Use o cenário mais conservador ({risco_cons}/5) e sinalize a incerteza no relatório. "
            f"Confiança ensemble: {conf_ens}%."
        )
    elif conv_status == "parcial":
        conv_desc = (
            f"ANÁLISE PARCIAL — apenas um especialista respondeu. "
            f"Risco consolidado: {risco_cons}/5. Confiança penalizada: {conf_ens}% (sem validação cruzada)."
        )
    else:
        conv_desc = "Nenhuma análise especializada disponível — baseie-se apenas nos dados brutos."

    prob = estado.get("probabilidades_previsao", {})
    prob_txt = ""
    if prob:
        prob_txt = (
            f"P(<30%) h+7: {prob.get('prob_sub_30_pct_h7', 'N/A')}% | "
            f"P(<50%) h+7: {prob.get('prob_sub_50_pct_h7', 'N/A')}% | "
            f"P(>80%) h+7: {prob.get('prob_acima_80_pct_h7', 'N/A')}% | "
            f"p10/p90 h+1: {prob.get('p10_h1', 'N/A')}%/{prob.get('p90_h1', 'N/A')}%"
        )

    prob_hist_dia = estado.get("prob_hist_dia") or {}
    prob_hist_fut = estado.get("prob_hist_futuro") or {}
    prob_hist_txt = ""
    if prob_hist_dia or prob_hist_fut:
        linhas = []
        if prob_hist_dia:
            for sistema, info in prob_hist_dia.items():
                linhas.append(
                    f"{sistema} (doy={info.get('doy')}): p10={info.get('p10')} p50={info.get('p50')} p90={info.get('p90')}"
                )
        if prob_hist_fut:
            for sistema, offs in prob_hist_fut.items():
                for off, info in sorted(offs.items()):
                    linhas.append(
                        f"{sistema} h+{off}d (mes {info.get('mes_ref')}): "
                        f"p10={info.get('p10')} p50={info.get('p50')} p90={info.get('p90')}"
                    )
        prob_hist_txt = "\n## HISTÓRICO EMPÍRICO\n" + "\n".join(linhas)

    anomalias_txt = ""
    anomalias = estado.get("anomalias_historicas") or {}
    if anomalias:
        partes = []
        for nome, info in anomalias.items():
            partes.append(
                f"{nome}: z={info.get('zscore', 0):+.2f} "
                f"(atual={info.get('valor')}%, p10={info.get('p10_mes')}%, p90={info.get('p90_mes')}%)"
            )
        anomalias_txt = "\n## ANOMALIAS HISTÓRICAS (volume atual vs climatologia)\n" + "\n".join(partes)

    # Contexto sazonal
    mes_atual = datetime.now().month
    estacao = "PERÍODO SECO (JAS)" if 7 <= mes_atual <= 9 else "PERÍODO CHUVOSO (DJF)" if mes_atual in [12, 1, 2] else "TRANSIÇÃO"

    return f"""Você é o HydraIA, sistema de análise de risco hídrico da Macrometrópole de São Paulo.

Você recebeu análises de dois especialistas independentes. Sua tarefa é sintetizá-las em um relatório técnico final.

## CONTEXTO CLIMATOLÓGICO
Mês atual: {mes_atual} | Estação: {estacao}

## DADOS DOS RESERVATÓRIOS (SABESP)
{_formatar_sabesp(estado)}

## PROJEÇÃO HIDRO-LSTM
{_formatar_lstm(previsoes)}

## PREVISÃO METEOROLÓGICA (OpenWeather)
{_formatar_weather(weather, max_dias=len(previsoes))}

## ANÁLISES DOS ESPECIALISTAS
{fmt_analise(analise_hidro, "Especialista Hidrológico")}

{fmt_analise(analise_meteo, "Especialista Meteorológico")}

## VALIDAÇÃO CRUZADA
{conv_desc}

## PROBABILIDADES (HydroLSTM + resíduos)
{prob_txt}

{prob_hist_txt}

{anomalias_txt}

## CONTEXTO CIENTÍFICO (TCC HydraIA)
{contexto[:2000]}

## TAREFA
As seções factuais 1 a 5 serão renderizadas deterministicamente pelo sistema.
Produza SOMENTE as seções interpretativas 6 a 8, sem repetir tabelas nem criar
novos números. Use apenas os valores fornecidos e deixe explícitas incertezas,
limitações, divergências entre horizontes e a natureza heurística da confiança.

### 6) Riscos e Convergência
- Risco hidrológico: ...
- Risco meteorológico: ...
- Risco consolidado: ...
- Convergência: ...
- Baseline vs LSTM: ...

### 7) Recomendações Operacionais
- Curto prazo (7d): ...
- Médio prazo (30-90d): ...

### 8) Insights e Validações
- Padrões históricos relevantes e impacto no cálculo atual.
- Análise das métricas agregadas (Média/STD).
- Sinais de alerta/inconsistência.

Não escreva conteúdo antes da seção 6 nem depois da seção 8."""


# ─── Orquestrador ─────────────────────────────────────────────────────────────

# Tipo do sintetizador: recebe (prompt, max_tokens) → texto final.
Sintetizador = Callable[[str, int], str]


class OrquestradorLLM:
    """Ensemble: hidro ‖ meteo → cross-validation → sintetizador local.

    Recebe os especialistas via Dependency Injection — testes podem substituir
    qualquer componente por mocks sem reescrever a lógica de cross-validation.
    """

    def __init__(
        self,
        hidro: LLMSpecialist,
        meteo: LLMSpecialist,
        sintetizador: Sintetizador,
    ) -> None:
        self.hidro = hidro
        self.meteo = meteo
        self.sintetizador = sintetizador
        logger.info(
            "Especialistas configurados — hidro: %s (%s) | meteo: %s (%s)",
            hidro.nome, hidro.model, meteo.nome, meteo.model,
        )

    # ── Factory de conveniência ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, *, strict_schema: bool = True) -> "OrquestradorLLM":
        """Constroi o ensemble padrao (Gemini hidro + GPT meteo + LLM local)
        lendo `config_apis.json` e `.env` com validacao de schema e fallbacks.

        Args:
            strict_schema: se True, respostas com campos obrigatorios faltando
                sao descartadas (vira `parcial` no cross-validation).
        """
        from src.core.config import carregar_config
        from src.core.validacao_semantica import CAMPOS_HIDRO, CAMPOS_METEO
        cfg = carregar_config()

        g = cfg["gemini_ultra"]
        gemini = GeminiSpecialist(
            api_key=EnvLoader.get("GEMINI_ULTRA_API_KEY"),
            model=g["model"],
            temperature=g.get("temperature", 0.3),
            max_tokens=g.get("max_tokens", 2048),
            max_prompt_chars=g.get("max_prompt_chars", 60_000),
            timeout_s=float(g.get("timeout", 45)),
            nome="Gemini",
            strict_schema=strict_schema,
            campos_obrigatorios=CAMPOS_HIDRO,
        )

        o = cfg["openai_gpt5"]
        gpt = GPTSpecialist(
            api_key=EnvLoader.get("OPENAI_GPT5_API_KEY"),
            model=o["model"],
            temperature=o.get("temperature", 0.3),
            max_tokens=o.get("max_tokens", 2048),
            max_prompt_chars=o.get("max_prompt_chars", 60_000),
            timeout_s=float(o.get("timeout", 45)),
            nome="GPT",
            strict_schema=strict_schema,
            campos_obrigatorios=CAMPOS_METEO,
        )
        # Lista de fallbacks acessivel pelo orquestrador
        gpt._fallback_models = list(o.get("fallback_models", []))
        gemini._fallback_models = list(g.get("fallback_models", []))

        return cls(hidro=gemini, meteo=gpt, sintetizador=ask_mistral)

    # ── Fase 1: análises paralelas ────────────────────────────────────────────

    def _analisar_paralelo(
        self,
        estado: Dict,
        previsoes: list,
        weather: Optional[Dict],
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Dispara os dois especialistas em paralelo (I/O-bound → ThreadPoolExecutor)."""
        prompt_h = _prompt_hidrologico_ext(estado, previsoes)
        prompt_m = _prompt_meteorologico(estado, previsoes, weather)

        with ThreadPoolExecutor(max_workers=2) as ex:
            with METRICS.timer("orquestrador.especialistas_paralelo"):
                fut_h = ex.submit(self.hidro.analisar, prompt_h)
                fut_m = ex.submit(self.meteo.analisar, prompt_m)
                analise_hidro = fut_h.result()
                analise_meteo = fut_m.result()
        METRICS.inc("orquestrador.hidro.ok" if analise_hidro else "orquestrador.hidro.fail")
        METRICS.inc("orquestrador.meteo.ok" if analise_meteo else "orquestrador.meteo.fail")

        if analise_hidro:
            logger.info("[%s] Risco: %s/5 | Tendência: %s | Confiança: %s%%",
                        self.hidro.nome,
                        analise_hidro.get("nivel_risco"),
                        analise_hidro.get("tendencia"),
                        analise_hidro.get("confianca_pct"))
        if analise_meteo:
            logger.info("[%s] Risco: %s/5 | Impacto: %s | Chuva esperada: %smm",
                        self.meteo.nome,
                        analise_meteo.get("nivel_risco"),
                        analise_meteo.get("impacto_nos_reservatorios"),
                        analise_meteo.get("chuva_total_esperada_mm"))

        return analise_hidro, analise_meteo

    # ── Fase 2: cross-validation ──────────────────────────────────────────────

    def _validar_convergencia(
        self,
        analise_hidro: Optional[Dict],
        analise_meteo: Optional[Dict],
    ) -> Dict:
        risco_h = analise_hidro.get("nivel_risco") if analise_hidro else None
        risco_m = analise_meteo.get("nivel_risco") if analise_meteo else None
        conf_h = analise_hidro.get("confianca_pct", 0) if analise_hidro else 0
        conf_m = analise_meteo.get("confianca_pct", 0) if analise_meteo else 0

        if risco_h is None and risco_m is None:
            return {"status": "sem_dados", "confianca_ensemble_pct": 0}

        if risco_h is None or risco_m is None:
            risco_unico = risco_h or risco_m
            conf_unica = conf_h or conf_m
            return {
                "status": "parcial",
                "risco_consolidado": risco_unico,
                "confianca_ensemble_pct": round(conf_unica * 0.7),
            }

        delta = abs(risco_h - risco_m)
        risco_consolidado = max(risco_h, risco_m)
        conf_base = round((conf_h + conf_m) / 2)

        if delta <= 1:
            conf_ensemble = min(100, round(conf_base * 1.15))
            status = "convergencia"
        else:
            penalizacao = delta * 10
            conf_ensemble = max(0, conf_base - penalizacao)
            status = "divergencia"

        # Convergencia semantica (delta de risco + tendencia/impacto + confianca)
        score_sem, obs_sem = convergencia_semantica(analise_hidro, analise_meteo)
        # Penaliza ate 20pp se o score_sem for baixo; bonifica ate +5pp se alto
        ajuste = round((score_sem - 0.6) * 25)   # score=0.6 → 0; score=1.0 → +10; score=0 → -15
        conf_ensemble_aj = max(0, min(100, conf_ensemble + ajuste))

        return {
            "status": status,
            "risco_hidro": risco_h,
            "risco_meteo": risco_m,
            "delta": delta,
            "risco_consolidado": risco_consolidado,
            "confianca_ensemble_pct": conf_ensemble_aj,
            "convergencia_semantica_score": round(score_sem, 3),
            "convergencia_semantica_obs": obs_sem,
        }

    # ── Fase 3: síntese local ─────────────────────────────────────────────────

    def _sintetizar(
        self,
        estado: Dict,
        previsoes: list,
        weather: Optional[Dict],
        contexto: str,
        analise_hidro: Optional[Dict],
        analise_meteo: Optional[Dict],
        convergencia: Dict,
    ) -> str:
        logger.info("[Sintetizador] Gerando relatório final...")
        prompt = _prompt_sintese(
            estado, previsoes, weather, contexto,
            analise_hidro, analise_meteo, convergencia,
        )
        narrativa = self.sintetizador(prompt, 2000)
        return _montar_relatorio_final(
            narrativa,
            estado,
            previsoes,
            weather,
            exigir_secoes=self.sintetizador is ask_mistral,
        )

    # ── Pipeline público ──────────────────────────────────────────────────────

    def gerar_relatorio(
        self,
        estado_atual: Dict,
        previsoes_ml: list,
        weather: Optional[Dict],
        contexto_cientifico: str,
    ) -> str:
        """Executa o pipeline ensemble completo e retorna o relatório final."""
        logger.info("[Orquestrador] Fase 1 — Analises especializadas em paralelo")
        analise_hidro, analise_meteo = self._analisar_paralelo(estado_atual, previsoes_ml, weather)

        # ── Fase 1b: validacao semantica + deteccao de alucinacao ──
        vol_pct = estado_atual.get("volume_cantareira_pct")
        analise_hidro, alertas_hidro = avaliar_analise(
            self.hidro.nome, analise_hidro, tipo="hidro", volume_pct=vol_pct,
        )
        analise_meteo, alertas_meteo = avaliar_analise(
            self.meteo.nome, analise_meteo, tipo="meteo",
        )

        logger.info("[Orquestrador] Fase 2 — Cross-validation")
        convergencia = self._validar_convergencia(analise_hidro, analise_meteo)
        convergencia["alertas_hidro"] = alertas_hidro
        convergencia["alertas_meteo"] = alertas_meteo
        logger.info("[Cross-validation] Status: %s | Risco consolidado: %s/5 | Confianca ensemble: %s%% | conv_sem=%s",
                    convergencia.get("status"),
                    convergencia.get("risco_consolidado", "N/A"),
                    convergencia.get("confianca_ensemble_pct", 0),
                    convergencia.get("convergencia_semantica_score", "N/A"))

        logger.info("[Orquestrador] Fase 3 — Sintese local")
        relatorio = self._sintetizar(
            estado_atual, previsoes_ml, weather, contexto_cientifico,
            analise_hidro, analise_meteo, convergencia,
        )

        self._salvar_relatorio(relatorio, estado_atual, convergencia, previsoes_ml, weather)
        return relatorio

    def _salvar_relatorio(
        self,
        relatorio: str,
        estado_atual: Dict,
        convergencia: Dict,
        previsoes_ml: list,
        weather: Optional[Dict]
    ) -> None:
        """Persiste o relatorio em storage/outputs/reports/ com timestamp + versionamento
        e salva dados estruturados no SQLite.
        """
        from src.utils.budget import BUDGET
        from src.utils.persistence import PersistenceManager
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta = HydraPaths.BASE / "storage" / "outputs" / "reports"
        pasta.mkdir(parents=True, exist_ok=True)

        caminho_md = pasta / f"relatorio_{ts}.md"
        caminho_md.write_text(relatorio, encoding="utf-8")

        # Hash do modelo LSTM (se existir) para versionamento
        modelo_hash = "n/a"
        try:
            pt = HydraPaths.TIME_SERIES / "hydro_lstm_melhor.pt"
            if pt.exists():
                import hashlib as _hl
                with open(pt, "rb") as f:
                    modelo_hash = _hl.sha256(f.read()).hexdigest()[:12]
        except OSError:
            pass

        # Extrai tempos por etapa do snapshot de metricas (somente latencias)
        snap = METRICS.snapshot()
        tempos_etapa = {
            k.replace(".latency_s", ""): v
            for k, v in snap["histograms"].items()
            if k.endswith(".latency_s")
        }

        metadados = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id(),
            "data_referencia": estado_atual.get("data_referencia"),
            "volume_cantareira_pct": estado_atual.get("volume_cantareira_pct"),
            "situacao_operacional": estado_atual.get("situacao_operacional"),
            "probabilidades_previsao": estado_atual.get("probabilidades_previsao"),
            "anomalias_historicas": estado_atual.get("anomalias_historicas"),
            "prob_hist_dia": estado_atual.get("prob_hist_dia"),
            "prob_hist_futuro": estado_atual.get("prob_hist_futuro"),
            "ensemble": {
                "status": convergencia.get("status"),
                "risco_consolidado": convergencia.get("risco_consolidado"),
                "confianca_ensemble_pct": convergencia.get("confianca_ensemble_pct"),
                "convergencia_semantica_score": convergencia.get("convergencia_semantica_score"),
                "alertas_hidro": convergencia.get("alertas_hidro", []),
                "alertas_meteo": convergencia.get("alertas_meteo", []),
            },
            "modelos": {
                "hidro": {
                    "nome": self.hidro.nome,
                    "model_configurado": self.hidro.model,
                    "model_efetivo": self.hidro.last_model_used,
                },
                "meteo": {
                    "nome": self.meteo.nome,
                    "model_configurado": self.meteo.model,
                    "model_efetivo": self.meteo.last_model_used,
                },
                "hydro_lstm_sha": modelo_hash,
            },
            "versao_codigo": versao_codigo(),  # hash de conteudo real
            "tempos_por_etapa_s": tempos_etapa,
            "orcamento": BUDGET.snapshot(),
            "metricas": snap,
            "modo_dados": estado_atual.get("modo_dados", "real"),
            "fonte_dados": estado_atual.get("fonte_dados"),
            "arquivo_relatorio": caminho_md.name,
        }
        caminho_json = pasta / f"metadata_{ts}.json"
        caminho_json.write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")

        # Persistência SQLite para Front-end
        db = PersistenceManager()
        db.salvar_execucao(relatorio, metadados, estado_atual, previsoes_ml, weather)

        logger.info("[Relatorio] Salvo em %s (versao codigo=%s, lstm_sha=%s, etapas=%d)",
                    caminho_md, metadados["versao_codigo"], modelo_hash, len(tempos_etapa))
