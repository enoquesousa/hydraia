import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import json
import pickle
from src.data.inferencia_ts import prever

class TestPhase4LSTMOpt(unittest.TestCase):

    @patch("src.data.inferencia_ts.MODELS_DIR")
    @patch("src.data.inferencia_ts.DATASETS_DIR")
    @patch("src.data.inferencia_ts._carregar_ultimos_dias")
    @patch("src.data.inferencia_ts.HydroLSTM")
    @patch("torch.load")
    @patch("torch.quantization.quantize_dynamic")
    @patch("torch.compile")
    def test_prever_otimizacao_chamada(self, mock_compile, mock_quant, mock_load, mock_lstm_class, mock_carregar, mock_datasets, mock_models):
        # Setup mocks
        mock_load.return_value = {}
        mock_lstm = mock_lstm_class.return_value
        mock_lstm.return_value = torch.zeros((1, 7, 1)) # Simular retorno do modelo

        mock_models.__truediv__.return_value.exists.return_value = True
        mock_models.__truediv__.return_value.read_text.return_value = json.dumps({
            "lookback": 10, "n_features": 5, "n_targets": 1, "horizon": 7,
            "hidden": 64, "n_layers": 1, "sistemas": ["Cantareira"]
        })

        mock_carregar.return_value = np.zeros((10, 5), dtype=np.float32)

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((10, 2), dtype=np.float32)
        mock_scaler.inverse_transform.return_value = np.zeros((7, 1), dtype=np.float32)

        # Mocking open/pickle
        with patch("builtins.open", MagicMock()):
            with patch("pickle.load", return_value=mock_scaler):
                # Forcar CPU para testar quantizacao
                with patch("torch.cuda.is_available", return_value=False):
                    prever(7, otimizar=True)

                    # Verificar se quantizacao foi tentada (em CPU)
                    self.assertTrue(mock_quant.called)
                    # Verificar se compile foi tentado (se disponivel no torch fake)
                    if hasattr(torch, "compile"):
                        self.assertTrue(mock_compile.called)

if __name__ == "__main__":
    unittest.main()
