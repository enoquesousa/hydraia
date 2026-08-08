"""Testes unitários para o OrquestradorLLM (cross-validation e extração JSON)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.orquestrador import _extrair_json, _validar_analise, OrquestradorLLM


class TestExtrairJson:
    """Testa a função de extração de JSON das respostas dos LLMs."""

    def test_extrai_json_de_bloco_markdown(self):
        texto = '```json\n{"nivel_risco": 3, "tendencia": "piora"}\n```'
        result = _extrair_json(texto)
        assert result == {"nivel_risco": 3, "tendencia": "piora"}

    def test_extrai_json_direto_no_texto(self):
        texto = 'Análise: {"nivel_risco": 2, "confianca_pct": 80} - fim'
        result = _extrair_json(texto)
        assert result["nivel_risco"] == 2
        assert result["confianca_pct"] == 80

    def test_retorna_none_para_json_invalido(self):
        texto = "Sem JSON aqui, apenas texto livre."
        result = _extrair_json(texto)
        assert result is None

    def test_extrai_json_sem_marcador_de_linguagem(self):
        texto = "```\n{\"nivel_risco\": 4}\n```"
        result = _extrair_json(texto)
        assert result == {"nivel_risco": 4}


class TestValidarAnalise:
    """Testa a validação dos JSONs retornados pelos LLMs."""

    def test_aceita_nivel_risco_int_valido(self):
        resultado = _validar_analise({"nivel_risco": 3, "confianca_pct": 80}, "Teste")
        assert resultado["nivel_risco"] == 3

    def test_converte_float_para_int(self):
        resultado = _validar_analise({"nivel_risco": 3.0, "confianca_pct": 80}, "Teste")
        assert resultado["nivel_risco"] == 3

    def test_rejeita_nivel_risco_string_nao_numerica(self):
        resultado = _validar_analise({"nivel_risco": "alto", "confianca_pct": 80}, "Teste")
        assert resultado is None

    def test_rejeita_nivel_risco_zero(self):
        resultado = _validar_analise({"nivel_risco": 0, "confianca_pct": 80}, "Teste")
        assert resultado is None

    def test_rejeita_nivel_risco_seis(self):
        resultado = _validar_analise({"nivel_risco": 6, "confianca_pct": 80}, "Teste")
        assert resultado is None

    def test_clamp_confianca_acima_de_100(self):
        resultado = _validar_analise({"nivel_risco": 2, "confianca_pct": 150}, "Teste")
        assert resultado["confianca_pct"] == 100

    def test_confianca_ausente_vira_zero(self):
        resultado = _validar_analise({"nivel_risco": 2}, "Teste")
        assert resultado["confianca_pct"] == 0

    def test_rejeita_none(self):
        assert _validar_analise(None, "Teste") is None

    def test_rejeita_dict_vazio(self):
        assert _validar_analise({}, "Teste") is None


class TestCrossValidation:
    """Testa a lógica de cross-validation do orquestrador."""

    def _criar_orquestrador(self):
        """Instância sem carregar config/chaves reais."""
        obj = object.__new__(OrquestradorLLM)
        return obj

    def test_convergencia_delta_zero(self):
        orc = self._criar_orquestrador()
        hidro = {"nivel_risco": 3, "confianca_pct": 80}
        meteo = {"nivel_risco": 3, "confianca_pct": 90}
        result = orc._validar_convergencia(hidro, meteo)
        assert result["status"] == "convergencia"
        assert result["risco_consolidado"] == 3
        assert result["confianca_ensemble_pct"] > 80  # bonificado

    def test_convergencia_delta_um(self):
        orc = self._criar_orquestrador()
        hidro = {"nivel_risco": 3, "confianca_pct": 80}
        meteo = {"nivel_risco": 4, "confianca_pct": 80}
        result = orc._validar_convergencia(hidro, meteo)
        assert result["status"] == "convergencia"
        assert result["risco_consolidado"] == 4  # princípio conservador

    def test_divergencia_delta_dois(self):
        orc = self._criar_orquestrador()
        hidro = {"nivel_risco": 2, "confianca_pct": 80}
        meteo = {"nivel_risco": 4, "confianca_pct": 80}
        result = orc._validar_convergencia(hidro, meteo)
        assert result["status"] == "divergencia"
        assert result["delta"] == 2
        assert result["confianca_ensemble_pct"] < 80  # penalizado

    def test_parcial_apenas_hidro(self):
        orc = self._criar_orquestrador()
        hidro = {"nivel_risco": 3, "confianca_pct": 70}
        result = orc._validar_convergencia(hidro, None)
        assert result["status"] == "parcial"
        assert result["confianca_ensemble_pct"] == round(70 * 0.7)

    def test_parcial_apenas_meteo(self):
        orc = self._criar_orquestrador()
        meteo = {"nivel_risco": 5, "confianca_pct": 60}
        result = orc._validar_convergencia(None, meteo)
        assert result["status"] == "parcial"
        assert result["risco_consolidado"] == 5

    def test_sem_dados_ambos_none(self):
        orc = self._criar_orquestrador()
        result = orc._validar_convergencia(None, None)
        assert result["status"] == "sem_dados"
        assert result["confianca_ensemble_pct"] == 0

    def test_risco_consolidado_e_conservador(self):
        """Sempre usa o risco mais alto entre os dois especialistas."""
        orc = self._criar_orquestrador()
        hidro = {"nivel_risco": 2, "confianca_pct": 90}
        meteo = {"nivel_risco": 3, "confianca_pct": 90}
        result = orc._validar_convergencia(hidro, meteo)
        assert result["risco_consolidado"] == 3
