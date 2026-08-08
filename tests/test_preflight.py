"""Testes do modulo de pre-flight (validacao rigida do ambiente)."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core import preflight


# ─── _api_key_valida ─────────────────────────────────────────────────────────

class TestApiKeyValida:
    def test_aceita_chave_real(self):
        assert preflight._api_key_valida("sk-1234567890abcdef", "OPENAI_GPT5_API_KEY")

    def test_rejeita_none(self):
        assert not preflight._api_key_valida(None, "X")

    def test_rejeita_string_vazia(self):
        assert not preflight._api_key_valida("", "X")

    def test_rejeita_curta(self):
        assert not preflight._api_key_valida("abc", "X")

    def test_rejeita_placeholder(self):
        assert not preflight._api_key_valida("your_api_key_here", "X")

    def test_rejeita_changeme(self):
        assert not preflight._api_key_valida("changeme", "X")


# ─── validar_ou_falhar ───────────────────────────────────────────────────────

class TestValidarOuFalhar:
    def test_passa_quando_tudo_ok(self):
        rs = [
            preflight.CheckResult("a", True, "ok", obrigatorio=True),
            preflight.CheckResult("b", True, "ok", obrigatorio=False),
        ]
        # Nao deve levantar
        preflight.validar_ou_falhar(rs, strict=True)

    def test_falha_em_strict_se_obrigatorio_falha(self):
        rs = [
            preflight.CheckResult("a", False, "erro", obrigatorio=True),
            preflight.CheckResult("b", True, "ok"),
        ]
        with pytest.raises(preflight.PreflightError):
            preflight.validar_ou_falhar(rs, strict=True)

    def test_permissivo_nao_falha(self):
        rs = [preflight.CheckResult("a", False, "erro", obrigatorio=True)]
        # Em modo permissivo: apenas warning
        preflight.validar_ou_falhar(rs, strict=False)

    def test_opcionais_nao_quebram_strict(self):
        rs = [
            preflight.CheckResult("opc", False, "off", obrigatorio=False),
            preflight.CheckResult("ok",  True,  "ok",  obrigatorio=True),
        ]
        preflight.validar_ou_falhar(rs, strict=True)


# ─── _check_lm_studio ────────────────────────────────────────────────────────

class TestCheckLmStudio:
    @patch("src.core.preflight.requests.get")
    def test_online_com_chat_model(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": [{"id": "qwen-2.5-14b"}]}
        mock_get.return_value = resp
        r = preflight._check_lm_studio()
        assert r.ok
        assert "qwen-2.5-14b" in r.detalhes["modelos"]

    @patch("src.core.preflight.requests.get")
    def test_so_embed_falha(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": [{"id": "nomic-embed-text"}]}
        mock_get.return_value = resp
        r = preflight._check_lm_studio()
        assert not r.ok
        assert "sem modelo de chat" in r.mensagem

    @patch("src.core.preflight.requests.get")
    def test_offline_falha(self, mock_get):
        import requests as _req
        mock_get.side_effect = _req.exceptions.ConnectionError("offline")
        r = preflight._check_lm_studio()
        assert not r.ok
        assert "inacessivel" in r.mensagem


# ─── relatorio_dict ──────────────────────────────────────────────────────────

class TestRelatorioDict:
    def test_serializavel(self):
        rs = [
            preflight.CheckResult("a", True, "ok", obrigatorio=True),
            preflight.CheckResult("b", False, "fail", obrigatorio=False),
        ]
        d = preflight.relatorio_dict(rs)
        assert d["ok"] is True  # b nao e obrigatorio
        assert len(d["checks"]) == 2
        assert d["checks"][0]["nome"] == "a"
