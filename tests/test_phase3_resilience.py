import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.core.integracao import (
    executar_pipeline_integrado,
    _baseline_previsoes,
    coletar_dados_sabesp_reais
)
from src.utils.paths import HydraPaths
import os
import tempfile
from pathlib import Path

class TestPhase3Resilience(unittest.TestCase):

    @patch("src.api.sabesp.SabespCollector.coletar_macrometropole_completa")
    def test_sabesp_fallback_snapshot(self, mock_coletar):
        # Simular falha da API mas sucesso no snapshot
        mock_coletar.return_value = {
            "sistemas": {"cantareira": {"volume_percentual": 33.3}},
            "modo": "snapshot"
        }

        # Limpar cache para forçar coleta
        from src.core.integracao import _cache_sabesp
        _cache_sabesp.invalidate("sabesp")

        dados = coletar_dados_sabesp_reais(allow_snapshot=True)
        self.assertEqual(dados["modo"], "snapshot")
        self.assertEqual(dados["sistemas"]["cantareira"]["volume_percentual"], 33.3)

    def test_baseline_climatologia(self):
        # Criar um arquivo NPZ temporario com media_mes
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ds_dir = tmp_path / "processed" / "datasets"
            ds_dir.mkdir(parents=True)

            # media_mes: 12 meses, 6 sistemas. Vamos colocar 42.0 no Cantareira (idx 0)
            media = np.zeros((12, 6), dtype=np.float32)
            media.fill(10.0)
            media[:, 0] = 42.0

            np.savez(ds_dir / "sabesp_dataset.npz", media_mes=media)

            with patch("src.utils.paths.HydraPaths.DATASETS", ds_dir):
                # Limpar cache SABESP para nao interferir
                from src.core.integracao import _cache_sabesp
                _cache_sabesp.invalidate("sabesp")

                prevs = _baseline_previsoes(7, usar_climatologia=True)
                self.assertEqual(prevs[0]["volume_cantareira_pct"], 42.0)
                self.assertEqual(prevs[0]["fonte_previsao"], "baseline_climatologia")

    @patch("src.core.integracao.carregar_contexto_markdown")
    @patch("src.core.integracao.gerar_previsoes_ml")
    @patch("src.core.integracao.coletar_dados_sabesp_reais")
    @patch("src.core.integracao.coletar_dados_openweather")
    def test_timing_logging(self, mock_ow, mock_sabesp, mock_ml, mock_ctx):
        mock_ctx.return_value = "contexto"
        mock_ml.return_value = [{"dia": 1, "volume_cantareira_pct": 50.0}]
        mock_sabesp.return_value = {
            "sistemas": {"cantareira": {"volume_percentual": 50.0}},
            "modo": "real"
        }
        mock_ow.return_value = {"previsoes": []}

        with self.assertLogs("src.core.integracao", level="INFO") as cm:
            executar_pipeline_integrado(Path("fake.md"), skip_llm=True)

            # Verificar se existem logs de tempo (contendo "s)")
            timing_logs = [l for l in cm.output if "s)" in l]
            self.assertTrue(len(timing_logs) >= 4)

    def test_markdown_resiliente(self):
        # Caminho que nao existe
        from src.core.integracao import carregar_contexto_markdown
        res = carregar_contexto_markdown(Path("arquivo_fantasma.md"))
        self.assertIn("contexto científico não carregado", res)

if __name__ == "__main__":
    unittest.main()
