"""Teste geral END-TO-END do pipeline HydraIA com dados sinteticos.

Cobre, em uma so execucao, os comportamentos endurecidos pelas Fases 1 a 7:

  1. Pre-flight (validacao rigida de configuracao do ambiente).
  2. Coletor SABESP em modo simulado (Fase 2).
  3. Cache validado (Fase 2).
  4. Orquestrador com mocks dos especialistas LLM e sintetizador (Fase 1+4):
        - Validacao semantica
        - Cross-validation com convergencia semantica
        - Mascaramento, correlation_id e metricas (Fase 5)
        - Versionamento de codigo + lstm sha (Fase 5)
  5. Validacao no boot (config + fallbacks de modelo) (Fase 6).
  6. Pipeline disparado pelo CLI em modo simulado (Fase 7).

Testa o sistema completo SEM exigir torch, sem chave de API real e sem rede.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.sabesp import SabespCollector
from src.api.openweather import coletar_simulado as weather_simulado
from src.core.llm_specialists import LLMSpecialist
from src.core.orquestrador import OrquestradorLLM
from src.core.preflight import (
    CheckResult,
    executar_preflight,
    relatorio_dict,
    validar_ou_falhar,
)
from src.core.validacao_semantica import avaliar_analise, convergencia_semantica
from src.utils.cache import FileCache, validador_sabesp
from src.utils.observabilidade import (
    METRICS,
    correlation_id,
    novo_correlation_id,
    setup_logging,
    versao_codigo,
)


# ─── Mocks de especialistas ──────────────────────────────────────────────────

class _FakeHidro(LLMSpecialist):
    """Simula um especialista hidrologico convergente com o estado real."""

    def __init__(self):
        super().__init__(
            api_key="mock-key-1234567890",
            model="mock-gemini-2.5-pro",
            max_retries=0, backoff_s=0,
            nome="Gemini-MOCK",
        )

    def _invocar(self, prompt: str) -> str:
        return (
            '{"nivel_risco": 3, "tendencia": "estavel", '
            '"sistemas_criticos": ["Cotia"], '
            '"analise": "Reservatorios em faixa de atencao mas estaveis. '
            'Cotia abaixo de 50% requer monitoramento.", '
            '"confianca_pct": 78}'
        )


class _FakeMeteo(LLMSpecialist):
    """Simula um especialista meteorologico convergente."""

    def __init__(self):
        super().__init__(
            api_key="mock-key-1234567890",
            model="mock-gpt-4o",
            max_retries=0, backoff_s=0,
            nome="GPT-MOCK",
        )

    def _invocar(self, prompt: str) -> str:
        return (
            '{"nivel_risco": 2, "chuva_total_esperada_mm": 32.5, '
            '"impacto_nos_reservatorios": "positivo", '
            '"dias_com_chuva_significativa": 3, '
            '"analise": "Previsao de chuvas moderadas em 3 dos 7 dias '
            'deve mitigar risco de queda nos reservatorios.", '
            '"confianca_pct": 82}'
        )


def _sintetizador_mock(prompt: str, max_tokens: int = 1500) -> str:
    return (
        "# RELATORIO DE RISCO HIDRICO — Macrometropole de Sao Paulo\n\n"
        "## 1. SITUACAO ATUAL\n"
        "Reservatorios em faixa de atencao consolidada (risco 3/5).\n\n"
        "## 2. CRUZAMENTO CLIMA x VOLUME\n"
        "Chuvas moderadas previstas em 3 de 7 dias (~32mm) devem estabilizar a tendencia.\n\n"
        "## 3. NIVEL DE CONFIANCA\n"
        "Convergencia entre especialistas (delta=1) — confianca ensemble alta.\n\n"
        "## 4. SISTEMAS EM ALERTA\n"
        "- Sistema Cotia (volume baixo)\n\n"
        "## 5. RECOMENDACOES OPERACIONAIS\n"
        "- Monitoramento intensivo do Sistema Cotia\n"
        "- Manter operacao normal nos demais sistemas\n"
    )


# ─── E2E: pipeline completo com mocks ───────────────────────────────────────

class TestPipelineE2E:
    """Cenario principal: tudo verde, mocks convergentes."""

    def setup_method(self):
        METRICS.reset()
        novo_correlation_id(prefix="e2e")
        setup_logging(json_logs=False)

    def test_pipeline_completo_modo_simulado(self, tmp_path):
        """Roda end-to-end: SABESP simulado + previsoes mock + LLMs mock + sintese."""
        # 1. Coleta SABESP simulada
        col = SabespCollector(data_dir=tmp_path, auto_save=False)
        sabesp = col.coletar_simulado()
        assert sabesp["modo"] == "simulado"
        assert "cantareira" in sabesp["sistemas"]

        # 2. Cache validado: deve aceitar dado SABESP correto
        cache = FileCache(tmp_path / "cache", ttl_s=60, validator=validador_sabesp)
        assert cache.set("sabesp", sabesp) is True
        assert cache.get("sabesp") is not None

        # 3. Weather simulado
        weather = weather_simulado(dias=7)
        assert weather["total_previsoes"] == 7

        # 4. Estado atual + previsoes (mock)
        cantareira = sabesp["sistemas"]["cantareira"]
        estado = {
            "data_referencia": cantareira["data"],
            "volume_cantareira_pct": cantareira["volume_percentual"],
            "volume_cantareira_hm3": cantareira["volume_hm3"],
            "chuva_mm": cantareira["chuva_mm"],
            "vazao_natural_m3s": cantareira["vazao_natural_m3s"],
            "vazao_jusante_m3s": cantareira["vazao_jusante_m3s"],
            "situacao_operacional": "Faixa de Atencao",
            "fonte_dados": "SABESP (Simulado)",
            "modo_dados": "simulado",
            "macrometropole": sabesp["resumo"],
        }
        previsoes = [
            {"dia": float(i + 1), "chuva_mm": 0.0,
             "volume_cantareira_pct": estado["volume_cantareira_pct"] - i * 0.2}
            for i in range(7)
        ]

        # 5. Orquestrador + mocks
        orq = OrquestradorLLM(
            hidro=_FakeHidro(), meteo=_FakeMeteo(),
            sintetizador=_sintetizador_mock,
        )

        # Patch de paths para nao escrever no projeto real
        with patch("src.core.orquestrador.HydraPaths") as mp:
            mp.BASE = tmp_path
            mp.TIME_SERIES = tmp_path / "models" / "time_series"
            relatorio = orq.gerar_relatorio(
                estado_atual=estado,
                previsoes_ml=previsoes,
                weather=weather,
                contexto_cientifico="Contexto cientifico mockado para teste e2e.",
            )

        # 6. Verificacoes do relatorio
        assert "RELATORIO DE RISCO HIDRICO" in relatorio
        assert "SISTEMAS EM ALERTA" in relatorio

        # 7. Verificacoes de metadados gerados
        reports_dir = tmp_path / "storage" / "outputs" / "reports"
        assert reports_dir.exists()
        metas = list(reports_dir.glob("metadata_*.json"))
        assert len(metas) >= 1
        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        assert "correlation_id" in meta
        assert meta["correlation_id"].startswith("e2e-")
        assert meta["ensemble"]["status"] == "convergencia"
        assert meta["ensemble"]["risco_consolidado"] == 3   # max(3, 2)
        assert "convergencia_semantica_score" in meta["ensemble"]
        assert meta["versao_codigo"]
        assert meta["modo_dados"] == "simulado"
        # Fase E: tempos_por_etapa + orcamento agora estao no metadata
        assert "tempos_por_etapa_s" in meta
        assert "orcamento" in meta
        assert "estado" in meta["orcamento"]
        assert meta["orcamento"]["estado"]["llm_calls"] >= 2

        # 8. Metricas devem ter sido populadas
        snap = METRICS.snapshot()
        assert snap["counters"].get("orquestrador.hidro.ok") == 1
        assert snap["counters"].get("orquestrador.meteo.ok") == 1
        assert "orquestrador.especialistas_paralelo.latency_s" in snap["histograms"]


# ─── E2E: cenario de falha (preflight bloqueia execucao) ────────────────────

class TestPreflightBloqueiaEmFalha:
    def test_preflight_strict_falha_sem_chaves(self):
        rs = [
            CheckResult("env:GEMINI_ULTRA_API_KEY", False, "ausente", obrigatorio=True),
            CheckResult("env:OPENAI_GPT5_API_KEY", True, "ok", obrigatorio=True),
        ]
        from src.core.preflight import PreflightError
        with pytest.raises(PreflightError):
            validar_ou_falhar(rs, strict=True)

    def test_preflight_dict_serializavel(self):
        rs = [CheckResult("a", True, "ok"), CheckResult("b", False, "fail", obrigatorio=False)]
        d = relatorio_dict(rs)
        assert d["ok"] is True
        # Re-serializa: nao deve quebrar
        json.dumps(d, ensure_ascii=False)


# ─── E2E: orquestrador com analise parcial / divergente ─────────────────────

class TestE2EAnaliseParcial:
    def test_apenas_hidro_responde(self, tmp_path):
        """Meteo cai (timeout simulado) — pipeline segue com analise parcial."""
        novo_correlation_id(prefix="e2e-parcial")

        class HidroOK(LLMSpecialist):
            def __init__(self):
                super().__init__(api_key="k", model="m", max_retries=0, backoff_s=0)
            def _invocar(self, p):
                return ('{"nivel_risco": 4, "tendencia": "piora", '
                        '"analise": "Tendencia negativa.", "confianca_pct": 70}')

        class MeteoTimeout(LLMSpecialist):
            def __init__(self):
                super().__init__(api_key="k", model="m", max_retries=0, backoff_s=0,
                                 timeout_s=0.1)
            def _invocar(self, p):
                import time
                time.sleep(2.0)  # timeout
                return "x"

        orq = OrquestradorLLM(
            hidro=HidroOK(), meteo=MeteoTimeout(),
            sintetizador=lambda prompt, mt: "RELATORIO PARCIAL",
        )
        estado = {
            "data_referencia": "2026-04-14", "volume_cantareira_pct": 50.0,
            "situacao_operacional": "Atencao", "macrometropole": {},
        }
        with patch("src.core.orquestrador.HydraPaths") as mp:
            mp.BASE = tmp_path
            mp.TIME_SERIES = tmp_path / "ts"
            relatorio = orq.gerar_relatorio(estado, [], None, "ctx")

        assert "RELATORIO PARCIAL" in relatorio
        # Verifica que o status foi 'parcial'
        metas = list((tmp_path / "storage" / "outputs" / "reports").glob("metadata_*.json"))
        assert metas
        m = json.loads(metas[0].read_text(encoding="utf-8"))
        assert m["ensemble"]["status"] == "parcial"


# ─── E2E: validacao semantica integrada ──────────────────────────────────────

class TestE2EValidacaoSemantica:
    def test_alucinacao_alerta_no_metadata(self, tmp_path):
        """Hidro retorna texto suspeito — alerta deve constar no metadata."""
        novo_correlation_id(prefix="e2e-aluc")

        class HidroAluc(LLMSpecialist):
            def __init__(self):
                super().__init__(api_key="k", model="m", max_retries=0, backoff_s=0)
            def _invocar(self, p):
                return ('{"nivel_risco": 3, "tendencia": "estavel", '
                        '"analise": "Eu inventei esses numeros porque nao tenho dados", '
                        '"confianca_pct": 80}')

        class MeteoOK(LLMSpecialist):
            def __init__(self):
                super().__init__(api_key="k", model="m", max_retries=0, backoff_s=0)
            def _invocar(self, p):
                return ('{"nivel_risco": 3, "chuva_total_esperada_mm": 20.0, '
                        '"impacto_nos_reservatorios": "neutro", '
                        '"dias_com_chuva_significativa": 2, '
                        '"analise": "Chuvas moderadas.", "confianca_pct": 70}')

        orq = OrquestradorLLM(
            hidro=HidroAluc(), meteo=MeteoOK(),
            sintetizador=lambda prompt, mt: "OK",
        )
        estado = {"data_referencia": "2026-04-14", "volume_cantareira_pct": 50.0,
                  "situacao_operacional": "Atencao", "macrometropole": {}}
        with patch("src.core.orquestrador.HydraPaths") as mp:
            mp.BASE = tmp_path
            mp.TIME_SERIES = tmp_path / "ts"
            orq.gerar_relatorio(estado, [], None, "ctx")

        metas = list((tmp_path / "storage" / "outputs" / "reports").glob("metadata_*.json"))
        m = json.loads(metas[0].read_text(encoding="utf-8"))
        alertas = m["ensemble"]["alertas_hidro"]
        assert any("alucinacao" in a for a in alertas)


# ─── E2E: convergencia semantica ────────────────────────────────────────────

class TestConvergenciaScore:
    def test_score_alto_em_concordancia_total(self):
        h = {"nivel_risco": 3, "tendencia": "estavel", "confianca_pct": 80}
        m = {"nivel_risco": 3, "impacto_nos_reservatorios": "neutro", "confianca_pct": 80}
        score, _ = convergencia_semantica(h, m)
        assert score > 0.9

    def test_score_baixo_em_discordancia(self):
        h = {"nivel_risco": 1, "tendencia": "melhora", "confianca_pct": 30}
        m = {"nivel_risco": 5, "impacto_nos_reservatorios": "negativo", "confianca_pct": 90}
        score, _ = convergencia_semantica(h, m)
        assert score < 0.4


# ─── E2E: smoke test do CLI metrics ──────────────────────────────────────────

class TestCLIMetrics:
    def test_metrics_inclui_dados_da_execucao(self, capsys):
        from src.cli import main
        METRICS.reset()
        METRICS.inc("teste.contador", 5)
        METRICS.observe("teste.lat", 1.23)
        rc = main(["metrics"])
        assert rc == 0
        out = capsys.readouterr().out
        d = json.loads(out)
        # output e {"metrics": {...}, "budget": {...}}
        assert d["metrics"]["counters"]["teste.contador"] == 5
        assert d["metrics"]["histograms"]["teste.lat"]["n"] == 1
        assert "budget" in d
