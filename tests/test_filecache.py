"""Testes do FileCache: TTL, escrita atômica e resiliência a corrupção."""

import json
import os
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.cache import FileCache


@pytest.fixture
def cache(tmp_path):
    return FileCache(tmp_path / "cache_dir", ttl_s=10)


class TestFileCacheBasico:
    def test_get_inexistente_retorna_none(self, cache):
        assert cache.get("nao_existe") is None

    def test_set_e_get_roundtrip(self, cache):
        cache.set("k", {"x": 42})
        assert cache.get("k") == {"x": 42}

    def test_invalidate_remove(self, cache):
        cache.set("k", {"x": 1})
        cache.invalidate("k")
        assert cache.get("k") is None


class TestFileCacheTTL:
    def test_expira_apos_ttl(self, cache):
        cache.set("k", {"x": 1})
        with patch("time.time", return_value=time.time() + 100):
            assert cache.get("k") is None

    def test_dentro_do_ttl(self, cache):
        cache.set("k", {"x": 1})
        assert cache.get("k") == {"x": 1}


class TestFileCacheAtomicidade:
    def test_escrita_nao_deixa_tmp_para_tras(self, cache):
        cache.set("k", {"x": 1})
        files = list(cache.dir.iterdir())
        # só deve existir k.json — nada de k.json.tmp
        assert all(not f.name.endswith(".tmp") for f in files)
        assert (cache.dir / "k.json").exists()

    def test_corrompido_e_invalidado(self, cache):
        # Simula corrupção: escreve JSON inválido direto
        path = cache.dir / "corrompido.json"
        path.write_text("isso nao eh JSON valido {{{", encoding="utf-8")
        # Ler deve retornar None E remover o arquivo (auto-cura)
        assert cache.get("corrompido") is None
        assert not path.exists()


class TestFileCacheConcorrencia:
    def test_dois_writes_seguidos_ultimo_vence(self, cache):
        cache.set("k", {"v": 1})
        cache.set("k", {"v": 2})
        assert cache.get("k") == {"v": 2}
