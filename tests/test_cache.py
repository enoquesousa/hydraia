"""Testes unitários para o sistema de cache de integracao.py."""

import json
import sys
import os
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.core.integracao as integracao


class TestCache:

    def _caminho_cache(self, nome):
        return integracao._CACHE_DIR / f"{nome}.json"

    def _escrever_cache(self, nome, dados):
        integracao._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._caminho_cache(nome).write_text(
            json.dumps(dados, ensure_ascii=False), encoding="utf-8"
        )

    def _remover_cache(self, nome):
        caminho = self._caminho_cache(nome)
        if caminho.exists():
            caminho.unlink()

    def test_ler_cache_retorna_none_se_nao_existe(self):
        self._remover_cache("_test_inexistente")
        assert integracao._ler_cache("_test_inexistente") is None

    def test_ler_cache_retorna_dados_validos(self):
        dados = {"sistemas": {"cantareira": {"volume_percentual": 43.0}}}
        self._escrever_cache("_test_valido", dados)
        resultado = integracao._ler_cache("_test_valido")
        assert resultado == dados
        self._remover_cache("_test_valido")

    def test_ler_cache_retorna_none_se_expirado(self):
        dados = {"sistemas": {}}
        self._escrever_cache("_test_expirado", dados)
        # Simula mtime 2 horas atrás
        with patch("time.time", return_value=time.time() + integracao._CACHE_TTL + 1):
            resultado = integracao._ler_cache("_test_expirado")
        assert resultado is None
        self._remover_cache("_test_expirado")

    def test_salvar_e_ler_cache_roundtrip(self):
        dados = {"sistemas": {"guarapiranga": {"volume_percentual": 89.2}}}
        integracao._salvar_cache("_test_roundtrip", dados)
        resultado = integracao._ler_cache("_test_roundtrip")
        assert resultado == dados
        self._remover_cache("_test_roundtrip")
