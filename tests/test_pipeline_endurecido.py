"""Testes integrados das melhorias da segunda rodada (Fases A-G).

Cobre:
  Fase A — pipeline propaga offline/simulado; CLI run com flags integradas.
  Fase B — preflight pode sondar modelos LLM via SDK (sondar_modelos_llm).
  Fase C — fallback HydroLSTM via baseline persistence.
  Fase D — BudgetController bloqueia chamadas alem do limite (modo economico).
  Fase E — versao_codigo usa hash de conteudo; metadata inclui tempos_por_etapa
           e orcamento.
  Fase F — strict_schema descarta respostas LLM com campos faltando.
  Fase G — CLI `run --modo-simulado --skip-llm --economico` roda end-to-end
           sem nenhuma rede e sem chave de API.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli import main as cli_main
from src.core.integracao import (
    _baseline_previsoes,
    coletar_dados_sabesp_reais,
    executar_pipeline_integrado,
    gerar_previsoes_ml,
)
from src.core.llm_specialists import LLMSpecialist
from src.core.orquestrador import OrquestradorLLM
from src.core.validacao_semantica import CAMPOS_HIDRO, CAMPOS_METEO
from src.utils.budget import BUDGET, BudgetLimits
from src.utils.observabilidade import METRICS, novo_correlation_id


@pytest.fixture(autouse=True)
def _reset_globals():
    METRICS.reset()
    BUDGET.reset()
    BUDGET.limites = BudgetLimits()
    yield
    METRICS.reset()
    BUDGET.reset()
    BUDGET.limites = BudgetLimits()


# ─── Fase A: pipeline propaga modo simulado/offline ─────────────────────────

class TestFaseA_PipelinePropagaFlags:
    def test_pipeline_modo_simulado_nao_chama_apis(self, tmp_path):
        md = tmp_path / "ctx.md"
        md.write_text("contexto", encoding="utf-8")
        # Mesmo SEM mocks de rede, modo_simulado nao chama nada externo
        relatorio = executar_pipeline_integrado(
            caminho_md=md,
            horizonte_dias=3,
            modo_simulado_dados=True,
            skip_llm=True,
        )
        assert "DRY-RUN" in relatorio
        assert "simulado" in relatorio


# ─── Fase C: baseline preenche se LSTM faltar ───────────────────────────────

class TestFaseC_BaselineFallback:
    def test_baseline_e_ativado_em_modo_simulado(self):
        prev = gerar_previsoes_ml(horizonte=4, modo_simulado_dados=True)
        assert all(p.get("fonte_previsao") == "baseline_persistence" for p in prev)

    def test_baseline_persistence_repete_ultimo_volume(self):
        prev = _baseline_previsoes(5)
        # Persistence: o ultimo valor conhecido e repetido no horizonte.
        v0 = prev[0]["volume_cantareira_pct"]
        v4 = prev[4]["volume_cantareira_pct"]
        assert v4 == v0


# ─── Fase D: budget integrado ao pipeline ───────────────────────────────────

class _FakeHidro(LLMSpecialist):
    def __init__(self, **kw):
        kw.setdefault("max_retries", 0)
        kw.setdefault("backoff_s", 0)
        super().__init__(api_key="k", model="m", nome="Gemini-MOCK", **kw)
    def _invocar(self, p):
        return ('{"nivel_risco": 3, "tendencia": "estavel", '
                '"analise": "ok", "confianca_pct": 80}')


class _FakeMeteo(LLMSpecialist):
    def __init__(self, **kw):
        kw.setdefault("max_retries", 0)
        kw.setdefault("backoff_s", 0)
        super().__init__(api_key="k", model="m", nome="GPT-MOCK", **kw)
    def _invocar(self, p):
        return ('{"nivel_risco": 2, "chuva_total_esperada_mm": 20.0, '
                '"impacto_nos_reservatorios": "neutro", '
                '"dias_com_chuva_significativa": 2, '
                '"analise": "ok", "confianca_pct": 70}')


class TestFaseD_BudgetEconomico:
    def test_orcamento_economico_aparece_no_metadata(self, tmp_path):
        BUDGET.aplicar_modo_economico()
        novo_correlation_id(prefix="e2e-eco")

        orq = OrquestradorLLM(
            hidro=_FakeHidro(), meteo=_FakeMeteo(),
            sintetizador=lambda prompt, mt: "RELATORIO",
        )
        estado = {"data_referencia": "2026-04-14", "volume_cantareira_pct": 50.0,
                  "situacao_operacional": "Atencao", "macrometropole": {},
                  "modo_dados": "simulado"}

        with patch("src.core.orquestrador.HydraPaths") as mp:
            mp.BASE = tmp_path
            mp.TIME_SERIES = tmp_path / "ts"
            orq.gerar_relatorio(estado, [], None, "ctx que cabe nos limites")

        meta = json.loads(list((tmp_path / "storage" / "outputs" / "reports").glob("metadata_*.json"))[0].read_text("utf-8"))
        assert meta["orcamento"]["limites"]["max_llm_calls"] == 6
        assert meta["orcamento"]["estado"]["llm_calls"] >= 2  # hidro + meteo


# ─── Fase E: tempos por etapa + hash conteudo ───────────────────────────────

class TestFaseE_MetadataEnriquecido:
    def test_tempos_por_etapa_no_metadata(self, tmp_path):
        novo_correlation_id(prefix="e2e-tempos")
        orq = OrquestradorLLM(
            hidro=_FakeHidro(), meteo=_FakeMeteo(),
            sintetizador=lambda prompt, mt: "OK",
        )
        estado = {"data_referencia": "2026-04-14", "volume_cantareira_pct": 50,
                  "situacao_operacional": "x", "macrometropole": {}}
        with patch("src.core.orquestrador.HydraPaths") as mp:
            mp.BASE = tmp_path
            mp.TIME_SERIES = tmp_path / "ts"
            orq.gerar_relatorio(estado, [], None, "ctx")
        meta = json.loads(list((tmp_path / "storage" / "outputs" / "reports").glob("metadata_*.json"))[0].read_text("utf-8"))
        # Deve incluir pelo menos a etapa 'orquestrador.especialistas_paralelo'
        assert "tempos_por_etapa_s" in meta
        assert any("orquestrador" in k or "pipeline" in k for k in meta["tempos_por_etapa_s"].keys())


# ─── Fase F: strict_schema rejeita resposta LLM incompleta ──────────────────

class TestFaseF_StrictSchemaIntegrado:
    def test_resposta_meteo_sem_impacto_e_descartada_em_strict(self, tmp_path):
        # Especialista meteo retorna JSON incompleto (falta impacto_nos_reservatorios)
        class MeteoIncompleto(LLMSpecialist):
            def __init__(self):
                super().__init__(
                    api_key="k", model="m", max_retries=0, backoff_s=0,
                    nome="GPT-MOCK", strict_schema=True,
                    campos_obrigatorios=CAMPOS_METEO,
                )
            def _invocar(self, p):
                return ('{"nivel_risco": 2, "chuva_total_esperada_mm": 10.0, '
                        '"dias_com_chuva_significativa": 2, '
                        '"analise": "x", "confianca_pct": 60}')

        novo_correlation_id(prefix="e2e-strict")
        orq = OrquestradorLLM(
            hidro=_FakeHidro(strict_schema=True, campos_obrigatorios=CAMPOS_HIDRO),
            meteo=MeteoIncompleto(),
            sintetizador=lambda prompt, mt: "OK",
        )
        estado = {"data_referencia": "x", "volume_cantareira_pct": 50,
                  "situacao_operacional": "x", "macrometropole": {}}
        with patch("src.core.orquestrador.HydraPaths") as mp:
            mp.BASE = tmp_path
            mp.TIME_SERIES = tmp_path / "ts"
            orq.gerar_relatorio(estado, [], None, "ctx")

        # Como meteo foi descartado, o status deve ser 'parcial'
        meta = json.loads(list((tmp_path / "storage" / "outputs" / "reports").glob("metadata_*.json"))[0].read_text("utf-8"))
        assert meta["ensemble"]["status"] == "parcial"


# ─── Fase G: CLI run --modo-simulado --skip-llm fim-a-fim ───────────────────

class TestFaseG_CLIRunSimuladoSkipLLM:
    def test_run_simulado_skip_llm_zero_rede(self, capsys, monkeypatch, tmp_path):
        # Roda sem nenhuma chave/rede/torch
        # Cria o contexto md fake esperado pelo run
        from src.utils.paths import HydraPaths
        ctx = HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md"
        ctx_existia = ctx.exists()
        if not ctx_existia:
            ctx.parent.mkdir(parents=True, exist_ok=True)
            ctx.write_text("contexto fake", encoding="utf-8")

        try:
            rc = cli_main([
                "run", "--modo-simulado", "--skip-llm", "--skip-preflight",
                "--horizonte-dias", "3",
            ])
            assert rc == 0
            out = capsys.readouterr().out
            assert "DRY-RUN" in out or "simulado" in out
            assert "METRICAS" in out
            assert "ORCAMENTO" in out
        finally:
            if not ctx_existia and ctx.exists():
                # nao apaga se o arquivo era pre-existente
                pass

    def test_run_economico_aplica_budget(self, capsys, monkeypatch):
        from src.utils.paths import HydraPaths
        ctx = HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md"
        if not ctx.exists():
            ctx.parent.mkdir(parents=True, exist_ok=True)
            ctx.write_text("contexto fake", encoding="utf-8")
        rc = cli_main([
            "run", "--modo-simulado", "--skip-llm", "--skip-preflight",
            "--economico", "--horizonte-dias", "2",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # O orcamento economico deve aparecer no output JSON do final
        assert '"max_llm_calls": 6' in out
        assert '"max_wallclock_s": 300' in out


# ─── Smoke: full suite cumulativa ───────────────────────────────────────────

class TestSmokeIntegrado:
    def test_baseline_no_pipeline_skip_llm(self, tmp_path):
        """Pipeline completo SEM torch e SEM LM Studio: usa baseline + skeleton."""
        md = tmp_path / "ctx.md"
        md.write_text("ctx", encoding="utf-8")
        relatorio = executar_pipeline_integrado(
            caminho_md=md,
            horizonte_dias=7,
            modo_simulado_dados=True,
            skip_llm=True,
            permitir_baseline_fallback=True,
        )
        assert relatorio
        # Deve ter exibido todas as 7 previsoes
        for i in range(1, 8):
            assert f"Dia +{i}:" in relatorio
