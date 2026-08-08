import unittest
from unittest.mock import patch, MagicMock
from src.core.preflight import (
    executar_preflight,
    _check_openweather_ping,
    _check_gemini_modelo,
    _check_openai_modelo,
    CheckResult
)
import requests

class TestPhase2Preflight(unittest.TestCase):

    @patch("requests.get")
    def test_openweather_ping_invalid_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        res = _check_openweather_ping("chave-invalida")
        self.assertFalse(res.ok)
        self.assertIn("chave invalida", res.mensagem)
        self.assertTrue(res.obrigatorio)

    @patch("requests.get")
    def test_openweather_ping_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        res = _check_openweather_ping("chave-valida")
        self.assertTrue(res.ok)
        self.assertIn("online", res.mensagem)

    @patch("google.genai.Client")
    def test_gemini_sondagem_fallback(self, mock_client_class):
        mock_client = mock_client_class.return_value
        m1 = MagicMock()
        m1.name = "models/gemini-pro-fallback"
        mock_client.models.list.return_value = [m1]

        res = _check_gemini_modelo("key", "gemini-ultra-missing", ["gemini-pro-fallback"])
        if not res.ok:
            print(f"\nDEBUG: {res.mensagem}")
        self.assertTrue(res.ok)
        self.assertIn("fallback 'gemini-pro-fallback' OK", res.mensagem)
        self.assertTrue(res.detalhes["fallback_em_uso"])

    @patch("openai.OpenAI")
    def test_openai_sondagem_fail_all(self, mock_openai_class):
        mock_client = mock_openai_class.return_value
        m1 = MagicMock()
        m1.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [m1]

        res = _check_openai_modelo("key", "gpt-5-missing", ["gpt-4-missing"])
        self.assertFalse(res.ok)
        self.assertIn("nenhum modelo acessivel", res.mensagem)

    @patch("src.core.preflight._check_chave")
    @patch("src.core.preflight._check_modelo_lstm")
    @patch("src.core.preflight._check_contexto_md")
    def test_executar_preflight_flow(self, mock_ctx, mock_lstm, mock_chave):
        mock_chave.return_value = CheckResult("chave", True)
        mock_lstm.return_value = CheckResult("lstm", True)
        mock_ctx.return_value = CheckResult("ctx", True)

        with patch("src.core.preflight._check_openweather_ping") as mock_ow:
            mock_ow.return_value = CheckResult("ow", True)
            res = executar_preflight(sondar_modelos_llm=False)

            # Verificar se os checks basicos estao la
            nomes = [r.nome for r in res]
            self.assertIn("ctx", nomes)
            self.assertIn("lstm", nomes)

if __name__ == "__main__":
    unittest.main()
