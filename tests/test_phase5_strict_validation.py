import unittest
from src.core.validacao_semantica import avaliar_analise

class TestPhase5StrictValidation(unittest.TestCase):

    def test_descarte_schema_incompleto(self):
        # Falta 'tendencia' e 'analise'
        analise_lixo = {
            "nivel_risco": 3,
            "confianca_pct": 80
        }
        res, alertas = avaliar_analise("Teste", analise_lixo, tipo="hidro")
        self.assertIsNone(res)
        self.assertTrue(any("schema: campos faltando" in a for a in alertas))

    def test_descarte_alucinacao(self):
        analise_alucinada = {
            "nivel_risco": 3,
            "tendencia": "estavel",
            "analise": "Eu inventei esses dados para o teste.",
            "confianca_pct": 100
        }
        res, alertas = avaliar_analise("Teste", analise_alucinada, tipo="hidro")
        self.assertIsNone(res)
        self.assertTrue(any("alucinacao" in a for a in alertas))

    def test_manter_analise_valida(self):
        analise_ok = {
            "nivel_risco": 2,
            "tendencia": "melhora",
            "analise": "Reservatorios subindo devido a chuvas.",
            "confianca_pct": 90
        }
        res, alertas = avaliar_analise("Teste", analise_ok, tipo="hidro")
        self.assertIsNotNone(res)
        self.assertEqual(len(alertas), 0)

if __name__ == "__main__":
    unittest.main()
