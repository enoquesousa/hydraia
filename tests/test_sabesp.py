"""Testes unitários para o coletor SABESP."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.sabesp import SabespCollector, ID_TO_SISTEMA, SISTEMAS_IDS


class TestSabespCollector:

    def test_id_to_sistema_contem_cantareira(self):
        """Cantareira deve estar mapeada no dicionário de IDs."""
        assert "cantareira" in ID_TO_SISTEMA.values()

    def test_id_to_sistema_e_inverso_de_sistemas_ids(self):
        """ID_TO_SISTEMA deve ser o inverso exato de SISTEMAS_IDS."""
        for chave, id_num in SISTEMAS_IDS.items():
            assert ID_TO_SISTEMA[id_num] == chave

    def test_normalizar_item_extrai_campos_corretos(self):
        """_normalizar_item deve extrair volume, chuva e vazões do JSON da API."""
        collector = SabespCollector()
        item = {
            "idSistema": 64,  # cantareira
            "volumeUtilArmazenadoPorcentagem": 43.69,
            "volumeUtilArmazenadoHm3": 789.5,
            "volumeTotalArmazenado": 901.2,
            "chuva": 1.2,
            "vazaoCaptada": 30.5,
            "vazaoProduzida": 28.0,
            "vazaoJusante": 5.0,
            "vazaoNatural": 25.0,
            "chuvaAcumuladaNoMes": 45.0,
        }
        result = collector._normalizar_item(item, "2026-04-14")
        assert result["volume_percentual"] == 43.69
        assert result["chuva_mm"] == 1.2
        assert result["vazao_natural_m3s"] == 25.0
        assert result["data"] == "2026-04-14"

    @patch("src.api.sabesp.requests.get")
    def test_obter_ultima_data_retorna_data(self, mock_get):
        """_obter_ultima_data deve retornar a data quando a API responde com sucesso."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"succeeded": True, "data": "2026-04-14"}
        mock_get.return_value = mock_resp

        collector = SabespCollector()
        data = collector._obter_ultima_data()
        assert data == "2026-04-14"

    @patch("src.api.sabesp.requests.get")
    def test_obter_ultima_data_retorna_none_em_erro(self, mock_get):
        """_obter_ultima_data deve retornar None quando a API falha."""
        mock_get.side_effect = Exception("timeout")
        collector = SabespCollector()
        data = collector._obter_ultima_data()
        assert data is None

    @patch("src.api.sabesp.requests.get")
    def test_coletar_macrometropole_completa(self, mock_get):
        """Pipeline completo deve retornar estrutura com 'sistemas' e 'resumo'."""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "ultima-data" in url:
                resp.json.return_value = {"succeeded": True, "data": "2026-04-14"}
            else:
                resp.json.return_value = {
                    "data": [
                        {
                            "idSistema": 64,
                            "volumeUtilArmazenadoPorcentagem": 43.69,
                            "volumeUtilArmazenadoHm3": 789.5,
                            "volumeTotalArmazenado": 901.2,
                            "chuva": 1.2,
                            "vazaoCaptada": 30.5,
                            "vazaoProduzida": 28.0,
                            "vazaoJusante": 5.0,
                            "vazaoNatural": 25.0,
                            "chuvaAcumuladaNoMes": 45.0,
                        }
                    ]
                }
            return resp

        mock_get.side_effect = side_effect
        collector = SabespCollector(auto_save=False)
        dados = collector.coletar_macrometropole_completa()

        assert "sistemas" in dados
        assert "cantareira" in dados["sistemas"]
        assert "resumo" in dados
        assert dados["sistemas"]["cantareira"]["volume_percentual"] == 43.69
