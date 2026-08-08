import unittest
import argparse
from unittest.mock import patch, MagicMock
from src.cli import _build_parser, main, cmd_run
from src.utils.observabilidade import correlation_id, METRICS, versao_codigo
from src.utils.paths import HydraPaths
import os
import tempfile
from pathlib import Path

class TestPhase1UXBoot(unittest.TestCase):
    def test_cli_flags_run(self):
        parser = _build_parser()
        # Test aliases and new flags
        args = parser.parse_args(["--json-logs", "run", "--allow-snapshot", "--dry-run", "--cid", "meu-cid-123"])
        self.assertTrue(args.allow_snapshot)
        self.assertTrue(args.dry_run)
        self.assertTrue(args.json_logs)
        self.assertEqual(args.cid, "meu-cid-123")

    def test_cid_customizado(self):
        # Simular execucao do cmd_run com CID customizado
        args = argparse.Namespace(
            cid="custom-cid",
            economico=False,
            dry_run=False,
            allow_snapshot=False,
            modo_simulado=True,
            sem_baseline_fallback=False,
            sondar_modelos=False,
            skip_preflight=True,
            horizonte_dias=7
        )

        with patch("src.core.integracao.executar_pipeline_integrado") as mock_pipe:
            mock_pipe.return_value = "Relatorio Mock"
            cmd_run(args)
            self.assertEqual(correlation_id(), "custom-cid")

    def test_metrics_reset_no_boot(self):
        METRICS.inc("teste.contagem", 10)
        self.assertEqual(METRICS.snapshot()["counters"]["teste.contagem"], 10)

        # Simular cmd_run que deve resetar
        args = argparse.Namespace(
            cid=None,
            economico=False,
            dry_run=True,
            allow_snapshot=True,
            modo_simulado=True,
            sem_baseline_fallback=False,
            sondar_modelos=False,
            skip_preflight=True,
            horizonte_dias=7
        )
        with patch("src.core.integracao.executar_pipeline_integrado"):
            cmd_run(args)

        self.assertEqual(METRICS.snapshot()["counters"], {})

    def test_versao_codigo_conteudo(self):
        """Verifica se o hash muda quando o conteudo de um arquivo muda."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            f1 = tmp_path / "file1.py"
            f1.write_text("print('hello')", encoding="utf-8")

            v1 = versao_codigo(tmp_path)

            # Muda conteudo
            f1.write_text("print('hello world')", encoding="utf-8")
            # Forcar limpeza do cache de 5s para o teste
            from src.utils.observabilidade import _VERSAO_CACHE
            _VERSAO_CACHE.clear()

            v2 = versao_codigo(tmp_path)
            self.assertNotEqual(v1, v2)

if __name__ == "__main__":
    unittest.main()
