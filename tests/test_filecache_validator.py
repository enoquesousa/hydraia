"""Testes do FileCache com validator: dados invalidos sao rejeitados."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.cache import FileCache, validador_sabesp, validador_openweather


@pytest.fixture
def cache_validado(tmp_path):
    return FileCache(tmp_path / "c", ttl_s=60, validator=validador_sabesp)


class TestSetValidator:
    def test_set_aceita_dado_valido(self, cache_validado):
        ok = cache_validado.set("k", {"sistemas": {"cantareira": {"volume_percentual": 50}}})
        assert ok is True
        assert cache_validado.get("k") is not None

    def test_set_rejeita_dado_invalido(self, cache_validado):
        ok = cache_validado.set("k", {"foo": "bar"})
        assert ok is False
        assert cache_validado.get("k") is None

    def test_set_rejeita_sistemas_vazio(self, cache_validado):
        assert not cache_validado.set("k", {"sistemas": {}})

    def test_set_rejeita_sem_volume(self, cache_validado):
        assert not cache_validado.set("k", {"sistemas": {"a": {}}})


class TestGetValidator:
    def test_get_invalida_arquivo_corrompido_logico(self, tmp_path):
        # Bypass do set: escreve direto no disco com dado invalido
        cache = FileCache(tmp_path / "c", ttl_s=60, validator=validador_sabesp)
        path = tmp_path / "c" / "x.json"
        path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        # get deve invalidar e remover
        assert cache.get("x") is None
        assert not path.exists()


class TestValidadorOpenWeather:
    def test_aceita_previsoes_lista(self):
        assert validador_openweather({"previsoes": [{"a": 1}]})

    def test_rejeita_previsoes_vazio(self):
        assert not validador_openweather({"previsoes": []})

    def test_rejeita_sem_previsoes(self):
        assert not validador_openweather({"foo": "bar"})

    def test_rejeita_nao_dict(self):
        assert not validador_openweather([])
