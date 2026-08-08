"""Estatísticas probabilísticas históricas para o HydraIA.

- Lê as séries brutas da SABESP armazenadas em `storage/raw/sabesp/series/sistemas`.
- Calcula estatísticas mensais (média, desvio-padrão, p10/p50/p90).
- Compara um valor atual contra a climatologia do mesmo mês.

Uso:
    from src.data.estatisticas_prob import (
        estatisticas_mensais,
        anomalias_volume_atual,
        comparar_valor_atual,
    )
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.constantes import PARAMS, SISTEMAS
from src.utils.paths import HydraPaths

# Mapeia slug SABESP → nome canônico usado no dataset/processamento.
_SABESP_TO_CANONICO: Dict[str, str] = {
    "cantareira": "Cantareira",
    "alto_tiete": "Alto Tietê",
    "guarapiranga": "Guarapiranga",
    "cotia": "Cotia",
    "rio_grande": "Rio Grande",
    "rio_claro": "Rio Claro",
    "sao_lourenco": "São Lourenço",
}

_TARGET_PARAM = "v_util"


def _carregar_serie(el_dir: Path, param_arquivo: str) -> pd.Series:
    """Lê um arquivo JSON de série temporal e retorna uma Series indexada por data."""
    path = el_dir / f"{param_arquivo}.json"
    if not path.exists():
        return pd.Series(dtype=float)

    raw = json.loads(path.read_text(encoding="utf-8"))
    registros = raw.get("dados", [])
    if not registros:
        return pd.Series(dtype=float)

    df = pd.DataFrame(registros)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")["value"]
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


@lru_cache(maxsize=1)
def _carregar_dataframe_bruto() -> pd.DataFrame:
    """Carrega todas as séries brutas disponíveis em um DataFrame MultiIndex."""
    series: Dict[tuple[str, str], pd.Series] = {}
    base_dir = HydraPaths.SABESP / "series" / "sistemas"

    for el_id, el_nome in SISTEMAS.items():
        el_dir = base_dir / el_id
        for param_arq, param_nome in PARAMS.items():
            s = _carregar_serie(el_dir, param_arq)
            if s.empty:
                continue
            series[(el_nome, param_nome)] = s

    if not series:
        return pd.DataFrame()

    df = pd.DataFrame(series)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["sistema", "parametro"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _series_por_param(parametro: str) -> Dict[str, pd.Series]:
    """Retorna um dicionário {sistema: series} já alinhado por data."""
    df = _carregar_dataframe_bruto()
    if df.empty or parametro not in df.columns.get_level_values("parametro"):
        return {}
    out: Dict[str, pd.Series] = {}
    for sistema in df.columns.get_level_values("sistema").unique():
        col = (sistema, parametro)
        if col in df.columns:
            s = df[col].dropna().sort_index()
            if not s.empty:
                out[sistema] = s
    return out


def estatisticas_mensais(parametro: str = _TARGET_PARAM) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Calcula estatísticas mensais por sistema para o parâmetro escolhido.

    Retorno:
        { sistema: { mes(int 1-12): {media,p10,p50,p90,std,n} } }
    """
    df = _carregar_dataframe_bruto()
    if df.empty or parametro not in df.columns.get_level_values("parametro"):
        return {}

    stats: Dict[str, Dict[int, Dict[str, float]]] = {}
    for sistema in sorted(df.columns.get_level_values("sistema").unique()):
        if (sistema, parametro) not in df.columns:
            continue
        serie = df[(sistema, parametro)].dropna()
        if serie.empty:
            continue
        tmp = serie.to_frame("valor")
        tmp["mes"] = tmp.index.month
        agg = tmp.groupby("mes")["valor"]
        stats[sistema] = {
            int(m): {
                "media": float(agg.mean().loc[m]),
                "p10": float(agg.quantile(0.10).loc[m]),
                "p50": float(agg.quantile(0.50).loc[m]),
                "p90": float(agg.quantile(0.90).loc[m]),
                "std": float(agg.std().loc[m]),
                "n": int(agg.count().loc[m]),
            }
            for m in agg.mean().index
        }
    return stats


def comparar_valor_atual(
    sistema: str,
    valor: float,
    *,
    parametro: str = _TARGET_PARAM,
    mes: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """Compara um valor observado com a climatologia do mês correspondente."""
    if valor is None:
        return None

    mes_ref = mes or date.today().month
    stats = estatisticas_mensais(parametro).get(sistema)
    if not stats or mes_ref not in stats:
        return None

    base = stats[mes_ref]
    std = base.get("std") or 0.0
    z = (valor - base.get("media", 0.0)) / std if std > 1e-6 else 0.0
    return {
        "valor": float(valor),
        "media_mes": base.get("media"),
        "p10_mes": base.get("p10"),
        "p90_mes": base.get("p90"),
        "zscore": float(z),
        "mes_ref": mes_ref,
    }


def anomalias_volume_atual(dados_sabesp: Dict) -> Dict[str, Dict[str, float]]:
    """Gera anomalias para o volume atual dos sistemas, comparando com a climatologia mensal."""
    sistemas = (dados_sabesp or {}).get("sistemas") or {}
    if not sistemas:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for slug, dados in sistemas.items():
        nome = _SABESP_TO_CANONICO.get(slug)
        if not nome:
            continue
        valor = dados.get("volume_percentual")
        comp = comparar_valor_atual(nome, valor, parametro=_TARGET_PARAM)
        if comp:
            out[nome] = comp
    return out


# ─── Estatísticas por dia-do-ano e previsão empírica ──────────────────────────


def estatisticas_por_dia_do_ano(
    parametro: str = _TARGET_PARAM,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Calcula quantis por dia-do-ano (1-366) para cada sistema."""
    series = _series_por_param(parametro)
    stats: Dict[str, Dict[int, Dict[str, float]]] = {}
    for sistema, s in series.items():
        tmp = s.to_frame("valor")
        tmp["doy"] = tmp.index.dayofyear
        grp = tmp.groupby("doy")["valor"]
        stats[sistema] = {
            int(d): {
                "p10": float(grp.quantile(0.10).loc[d]),
                "p50": float(grp.quantile(0.50).loc[d]),
                "p90": float(grp.quantile(0.90).loc[d]),
                "media": float(grp.mean().loc[d]),
                "n": int(grp.count().loc[d]),
            }
            for d in grp.mean().index
        }
    return stats


def previsao_empirica_meses(
    data_referencia: str | date | datetime,
    *,
    parametro: str = _TARGET_PARAM,
    offsets_dias: Sequence[int] = (1, 30, 90),
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Estimativa empírica para horizontes futuros usando climatologia mensal.

    Retorna {sistema: {offset: {mes_ref, p10,p50,p90,media}}}
    """
    if isinstance(data_referencia, str):
        data_ref_dt = datetime.fromisoformat(data_referencia).date()
    elif isinstance(data_referencia, datetime):
        data_ref_dt = data_referencia.date()
    else:
        data_ref_dt = data_referencia

    mensais = estatisticas_mensais(parametro)
    if not mensais:
        return {}

    resultados: Dict[str, Dict[int, Dict[str, float]]] = {}
    for sistema, por_mes in mensais.items():
        sis_res: Dict[int, Dict[str, float]] = {}
        for off in offsets_dias:
            alvo = data_ref_dt + timedelta(days=off)
            mes = alvo.month
            info = por_mes.get(mes)
            if not info:
                continue
            sis_res[off] = {
                "mes_ref": mes,
                "p10": info.get("p10"),
                "p50": info.get("p50"),
                "p90": info.get("p90"),
                "media": info.get("media"),
            }
        if sis_res:
            resultados[sistema] = sis_res
    return resultados


def probabilidade_dia_referencia(
    data_referencia: str | date | datetime,
    *,
    parametro: str = _TARGET_PARAM,
) -> Dict[str, Dict[str, float]]:
    """Quantis para o dia do ano da data de referência (base empírica)."""
    if isinstance(data_referencia, str):
        data_ref_dt = datetime.fromisoformat(data_referencia).date()
    elif isinstance(data_referencia, datetime):
        data_ref_dt = data_referencia.date()
    else:
        data_ref_dt = data_referencia

    doy_stats = estatisticas_por_dia_do_ano(parametro)
    if not doy_stats:
        return {}

    alvo = data_ref_dt.timetuple().tm_yday
    resultados: Dict[str, Dict[str, float]] = {}
    for sistema, stats in doy_stats.items():
        info = stats.get(alvo)
        if info:
            resultados[sistema] = {
                "doy": alvo,
                "p10": info.get("p10"),
                "p50": info.get("p50"),
                "p90": info.get("p90"),
                "media": info.get("media"),
            }
    return resultados
