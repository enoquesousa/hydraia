"""Validacao semantica e detec.de alucinacao para respostas dos especialistas.

Alem do nivel_risco basico, fazemos:
  * Validacao de schema completo (campos obrigatorios por especialista).
  * Sanity checks fisicos (chuva > 0, dias 0..7, etc.).
  * Coerencia interna (risco x tendencia, risco x estado dos reservatorios).
  * Detectores de alucinacao explicita (frase 'simulado', 'inventei', etc.).

Cada validacao retorna `(ok, motivos)` — `motivos` e uma lista de problemas
encontrados. Caller decide se descarta ou apenas penaliza confianca.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ResultadoValidacao:
    ok: bool
    motivos: List[str] = field(default_factory=list)
    severidade: str = "info"   # info | warn | erro

    def __bool__(self) -> bool:
        return self.ok


# ─── Schemas obrigatorios ────────────────────────────────────────────────────

CAMPOS_HIDRO = {
    "nivel_risco",
    "tendencia",
    "analise",
    "confianca_pct",
}

CAMPOS_METEO = {
    "nivel_risco",
    "chuva_total_esperada_mm",
    "impacto_nos_reservatorios",
    "dias_com_chuva_significativa",
    "analise",
    "confianca_pct",
}

VALORES_TENDENCIA = {"melhora", "estavel", "piora"}
VALORES_IMPACTO = {"positivo", "neutro", "negativo"}


def validar_schema_hidro(d: Dict) -> ResultadoValidacao:
    if not isinstance(d, dict):
        return ResultadoValidacao(False, ["nao e dict"], "erro")
    motivos: List[str] = []
    falt = CAMPOS_HIDRO - set(d.keys())
    if falt:
        motivos.append(f"campos faltando: {sorted(falt)}")
    tend = d.get("tendencia")
    if tend is not None and tend not in VALORES_TENDENCIA:
        motivos.append(f"tendencia invalida: {tend!r} (esperado {VALORES_TENDENCIA})")
    sis = d.get("sistemas_criticos")
    if sis is not None and not isinstance(sis, list):
        motivos.append(f"sistemas_criticos deveria ser lista")

    # Se faltar campo basico ou tendencia for lixo, e erro fatal para esta analise
    severidade = "erro" if (falt or (tend and tend not in VALORES_TENDENCIA)) else "info"
    return ResultadoValidacao(ok=not motivos, motivos=motivos, severidade=severidade)


def validar_schema_meteo(d: Dict) -> ResultadoValidacao:
    if not isinstance(d, dict):
        return ResultadoValidacao(False, ["nao e dict"], "erro")
    motivos: List[str] = []
    falt = CAMPOS_METEO - set(d.keys())
    if falt:
        motivos.append(f"campos faltando: {sorted(falt)}")
    imp = d.get("impacto_nos_reservatorios")
    if imp is not None and imp not in VALORES_IMPACTO:
        motivos.append(f"impacto invalido: {imp!r} (esperado {VALORES_IMPACTO})")

    severidade = "erro" if (falt or (imp and imp not in VALORES_IMPACTO)) else "info"
    return ResultadoValidacao(ok=not motivos, motivos=motivos, severidade=severidade)


# ─── Sanity checks fisicos ───────────────────────────────────────────────────

def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validar_fisica_meteo(d: Dict) -> ResultadoValidacao:
    """Detecta alucinacoes obvias na analise meteorologica."""
    motivos: List[str] = []
    chuva = _to_float(d.get("chuva_total_esperada_mm"))
    if chuva is not None:
        if chuva < 0:
            motivos.append(f"chuva negativa ({chuva}mm)")
        elif chuva > 1000:
            motivos.append(f"chuva absurda ({chuva}mm > 1000)")
    dias = d.get("dias_com_chuva_significativa")
    if dias is not None:
        di = _to_float(dias)
        if di is None or di < 0 or di > 14:
            motivos.append(f"dias_com_chuva_significativa invalido ({dias!r})")
    return ResultadoValidacao(ok=not motivos, motivos=motivos,
                              severidade="warn" if motivos else "info")


def validar_fisica_hidro(d: Dict, *, volume_pct: Optional[float] = None) -> ResultadoValidacao:
    """Risco deve ser razoavelmente coerente com o volume real do Cantareira.

    Heuristica conservadora — penaliza inconsistencias graves:
      * volume > 70% e risco >= 4 (alerta/critico): improvavel
      * volume < 20% e risco <= 2 (baixo/minimo): improvavel
    """
    motivos: List[str] = []
    risco = d.get("nivel_risco")
    if volume_pct is not None and isinstance(risco, int):
        if volume_pct > 70 and risco >= 4:
            motivos.append(f"volume {volume_pct}% mas risco={risco} (incoerente — volume alto)")
        elif volume_pct < 20 and risco <= 2:
            motivos.append(f"volume {volume_pct}% mas risco={risco} (incoerente — volume critico)")

    # Coerencia tendencia x risco
    tend = d.get("tendencia")
    if tend == "melhora" and isinstance(risco, int) and risco == 5:
        motivos.append("tendencia=melhora mas risco=5 (incoerente)")

    return ResultadoValidacao(ok=not motivos, motivos=motivos,
                              severidade="warn" if motivos else "info")


# ─── Deteccao de alucinacao textual ──────────────────────────────────────────

_PADROES_ALUCINACAO = [
    re.compile(r"\b(invent[a-z]+|simulei|simulad[ao]|fict[ií]cio|hipot[eé]tico)\b", re.IGNORECASE),
    re.compile(r"\b(n[aã]o (tenho|sei|posso))\b.*(dados|informacoes)", re.IGNORECASE),
]


def detectar_alucinacao_textual(d: Dict) -> ResultadoValidacao:
    """Procura pistas de alucinacao no campo 'analise'."""
    texto = d.get("analise", "") or ""
    motivos: List[str] = []
    for pat in _PADROES_ALUCINACAO:
        m = pat.search(texto)
        if m:
            motivos.append(f"texto sugere alucinacao: '{m.group(0)}'")
    return ResultadoValidacao(ok=not motivos, motivos=motivos,
                              severidade="warn" if motivos else "info")


# ─── Convergencia semantica ──────────────────────────────────────────────────

def convergencia_semantica(
    hidro: Optional[Dict],
    meteo: Optional[Dict],
) -> Tuple[float, List[str]]:
    """Pontuacao 0-1 de convergencia semantica entre hidro e meteo.

    Considera:
      * delta de nivel_risco (peso 50%)
      * coerencia tendencia/impacto (peso 30%)
      * delta de confianca (peso 20%)

    Retorna (pontuacao, observacoes).
    """
    if not hidro or not meteo:
        return 0.0, ["um dos especialistas nao retornou — sem convergencia semantica"]

    rh = hidro.get("nivel_risco")
    rm = meteo.get("nivel_risco")
    obs: List[str] = []

    if not (isinstance(rh, int) and isinstance(rm, int)):
        return 0.0, ["riscos nao numericos"]

    score_risco = max(0.0, 1.0 - abs(rh - rm) / 4.0)  # delta=0 -> 1.0; delta=4 -> 0.0
    obs.append(f"delta_risco={abs(rh-rm)} → score_risco={score_risco:.2f}")

    # Tendencia hidro vs impacto meteo
    tend = hidro.get("tendencia")
    imp = meteo.get("impacto_nos_reservatorios")
    pares_coerentes = {
        ("melhora", "positivo"),
        ("estavel", "neutro"),
        ("piora", "negativo"),
    }
    pares_compativeis = {
        ("estavel", "positivo"),
        ("estavel", "negativo"),
        ("melhora", "neutro"),
        ("piora", "neutro"),
    }
    if (tend, imp) in pares_coerentes:
        score_dir = 1.0
    elif (tend, imp) in pares_compativeis:
        score_dir = 0.6
    elif tend is None or imp is None:
        score_dir = 0.5
    else:
        score_dir = 0.0
    obs.append(f"tendencia={tend!r}/impacto={imp!r} → score_dir={score_dir:.2f}")

    # Confianca
    ch = float(hidro.get("confianca_pct", 0))
    cm = float(meteo.get("confianca_pct", 0))
    score_conf = max(0.0, 1.0 - abs(ch - cm) / 100.0)

    final = 0.5 * score_risco + 0.3 * score_dir + 0.2 * score_conf
    return float(final), obs


# ─── Validacao integrada ─────────────────────────────────────────────────────

def avaliar_analise(
    nome: str,
    d: Optional[Dict],
    *,
    tipo: str,                       # 'hidro' | 'meteo'
    volume_pct: Optional[float] = None,
) -> Tuple[Optional[Dict], List[str]]:
    """Pipeline de validacao semantica para uma analise.

    Aplica:
      * schema do tipo (erro fatal se faltar campo)
      * fisica do tipo (alertas)
      * deteccao de alucinacao textual (erro fatal se detectada)

    Retorna `(analise, alertas)`. Se houver erros criticos, retorna `None, alertas`.
    """
    if d is None:
        return None, ["analise ausente"]

    alertas: List[str] = []
    fatal = False

    if tipo == "hidro":
        r = validar_schema_hidro(d)
        if not r.ok:
            alertas.extend([f"schema: {m}" for m in r.motivos])
            if r.severidade == "erro": fatal = True
        f = validar_fisica_hidro(d, volume_pct=volume_pct)
        if not f.ok:
            alertas.extend([f"fisica: {m}" for m in f.motivos])
    elif tipo == "meteo":
        r = validar_schema_meteo(d)
        if not r.ok:
            alertas.extend([f"schema: {m}" for m in r.motivos])
            if r.severidade == "erro": fatal = True
        f = validar_fisica_meteo(d)
        if not f.ok:
            alertas.extend([f"fisica: {m}" for m in f.motivos])

    a = detectar_alucinacao_textual(d)
    if not a.ok:
        alertas.extend([f"alucinacao: {m}" for m in a.motivos])
        if a.severidade in ("erro", "warn"): # Alucinacao detectada e fatal por default
            fatal = True

    if alertas:
        log_level = logging.ERROR if fatal else logging.WARNING
        logger.log(log_level, "[%s] validacao semantica encontrou %d alerta(s) (fatal=%s): %s",
                       nome, len(alertas), fatal, alertas)

    if fatal:
        return None, alertas

    return d, alertas
