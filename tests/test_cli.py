"""Testes basicos do CLI: parser e subcomandos sem efeitos colaterais."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli import _build_parser, cmd_validate, main


# ─── Parser ──────────────────────────────────────────────────────────────────

class TestParser:
    def test_subcomandos_existem(self):
        p = _build_parser()
        # parsear deve funcionar para todos os subcomandos
        for cmd in ["validate", "dry-run", "run", "metrics", "preflight"]:
            args = p.parse_args([cmd])
            assert args.cmd == cmd

    def test_validate_flags(self):
        p = _build_parser()
        a = p.parse_args(["validate", "--strict", "--json", "--offline"])
        assert a.strict is True
        assert a.json is True
        assert a.offline is True

    def test_run_flags(self):
        p = _build_parser()
        a = p.parse_args(["run", "--modo-simulado", "--skip-preflight"])
        assert a.modo_simulado is True
        assert a.skip_preflight is True

    def test_run_todas_as_flags(self):
        p = _build_parser()
        a = p.parse_args([
            "run", "--offline", "--modo-simulado", "--skip-llm",
            "--economico", "--sem-baseline-fallback", "--sondar-modelos",
            "--skip-preflight", "--horizonte-dias", "5",
        ])
        assert a.allow_snapshot is True
        assert a.modo_simulado is True
        assert a.dry_run is True
        assert a.economico is True
        assert a.sem_baseline_fallback is True
        assert a.sondar_modelos is True
        assert a.skip_preflight is True
        assert a.horizonte_dias == 5

    def test_subcomando_obrigatorio(self):
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])


# ─── cmd_validate retorna codigo correto ────────────────────────────────────

class TestCmdValidate:
    def test_offline_mode_nao_exige_lm_studio(self):
        # Em modo offline, mesmo sem LM Studio o exit code pode ser 0
        # se as outras coisas estiverem ok. Usamos --offline + sem strict.
        ns = type("N", (), {"strict": False, "json": False, "offline": True})()
        with patch("src.core.preflight.executar_preflight") as mock_pre, \
             patch("src.core.preflight.imprimir_resumo"):
            from src.core.preflight import CheckResult
            mock_pre.return_value = [
                CheckResult("a", True, "ok", obrigatorio=True),
            ]
            assert cmd_validate(ns) == 0

    def test_strict_falha_retorna_1(self):
        ns = type("N", (), {"strict": True, "json": False, "offline": True})()
        with patch("src.core.preflight.executar_preflight") as mock_pre, \
             patch("src.core.preflight.imprimir_resumo"):
            from src.core.preflight import CheckResult
            mock_pre.return_value = [
                CheckResult("a", False, "fail", obrigatorio=True),
            ]
            assert cmd_validate(ns) == 1


# ─── cmd_metrics ─────────────────────────────────────────────────────────────

class TestCmdMetrics:
    def test_metrics_imprime_json(self, capsys):
        rc = main(["metrics"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "counters" in out
        assert "histograms" in out
