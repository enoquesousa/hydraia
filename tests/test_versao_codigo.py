"""Testes do hash de conteudo de codigo (anti-colisao por mtime/size)."""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.observabilidade import _VERSAO_CACHE, versao_codigo


@pytest.fixture(autouse=True)
def _limpar_cache():
    _VERSAO_CACHE.clear()
    yield
    _VERSAO_CACHE.clear()


class TestHashConteudo:
    def test_hash_e_deterministico(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("print('a')")
        b = tmp_path / "b.py"
        b.write_text("print('b')")
        h1 = versao_codigo(tmp_path)
        h2 = versao_codigo(tmp_path)
        assert h1 == h2

    def test_mudanca_de_conteudo_muda_hash(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("print('original')")
        h1 = versao_codigo(tmp_path)
        # Limpa cache para garantir releitura
        _VERSAO_CACHE.clear()
        a.write_text("print('alterado')")
        h2 = versao_codigo(tmp_path)
        assert h1 != h2

    def test_mtime_igual_conteudo_diferente_detecta(self, tmp_path):
        """Cenario que o hash mtime+size NAO detectaria — mas o de conteudo sim."""
        a = tmp_path / "a.py"
        a.write_text("xxxxxx")  # 6 chars
        st_antes = a.stat()
        h1 = versao_codigo(tmp_path, fast=False)
        # Sobrescreve mantendo o mesmo tamanho (6 chars) e fixa mtime
        _VERSAO_CACHE.clear()
        a.write_text("yyyyyy")  # 6 chars tambem
        os.utime(a, (st_antes.st_atime, st_antes.st_mtime))
        # fast=True (mtime+size) NAO detecta
        h_fast = versao_codigo(tmp_path, fast=True)
        # forca recalculo
        _VERSAO_CACHE.clear()
        h2_conteudo = versao_codigo(tmp_path, fast=False)
        # Default (fast=False) detecta a mudanca
        assert h1 != h2_conteudo

    def test_cache_evita_re_leitura(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("conteudo")
        h1 = versao_codigo(tmp_path)
        # Sem limpar cache: alterar arquivo nao muda hash dentro de 5s
        a.write_text("alterado")
        h2 = versao_codigo(tmp_path)
        assert h1 == h2  # cache em vigor

    def test_modo_fast_e_rapido_porem_aceita_colisao(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("xxxx")
        st = a.stat()
        h1 = versao_codigo(tmp_path, fast=True)
        _VERSAO_CACHE.clear()
        # Sobrescreve mantendo size; mexe so no conteudo
        a.write_text("yyyy")
        os.utime(a, (st.st_atime, st.st_mtime))
        h2 = versao_codigo(tmp_path, fast=True)
        # mtime+size: hash igual (caso colisao esperada)
        assert h1 == h2
