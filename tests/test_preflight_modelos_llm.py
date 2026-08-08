"""Testes da sondagem de modelos LLM no preflight (Gemini + OpenAI)."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core import preflight


# ─── Gemini ──────────────────────────────────────────────────────────────────

class TestCheckGeminiModelo:
    def test_sem_chave_falha(self):
        r = preflight._check_gemini_modelo(None, "gemini-2.5-pro")
        assert not r.ok

    def test_modelo_acessivel(self):
        fake_models = [type("M", (), {"name": "models/gemini-2.5-pro"})(),
                       type("M", (), {"name": "models/gemini-1.5-pro"})()]
        fake_client = MagicMock()
        fake_client.models.list.return_value = iter(fake_models)
        with patch.dict("sys.modules", {
            "google": MagicMock(genai=MagicMock(Client=lambda api_key: fake_client)),
            "google.genai": MagicMock(Client=lambda api_key: fake_client),
        }):
            r = preflight._check_gemini_modelo("k" * 20, "gemini-2.5-pro")
        assert r.ok
        assert "acessivel" in r.mensagem

    def test_modelo_inacessivel(self):
        fake_models = [type("M", (), {"name": "models/gemini-1.5-flash"})()]
        fake_client = MagicMock()
        fake_client.models.list.return_value = iter(fake_models)
        with patch.dict("sys.modules", {
            "google": MagicMock(genai=MagicMock(Client=lambda api_key: fake_client)),
            "google.genai": MagicMock(Client=lambda api_key: fake_client),
        }):
            r = preflight._check_gemini_modelo("k" * 20, "gemini-2.5-pro")
        assert not r.ok
        assert "nao acessivel" in r.mensagem


# ─── OpenAI ──────────────────────────────────────────────────────────────────

class TestCheckOpenaiModelo:
    def test_sem_chave_falha(self):
        r = preflight._check_openai_modelo(None, "gpt-4o", [])
        assert not r.ok

    def test_modelo_principal_acessivel(self):
        fake_models = MagicMock()
        fake_models.data = [type("M", (), {"id": "gpt-4o"})(), type("M", (), {"id": "gpt-3.5-turbo"})()]
        fake_client = MagicMock()
        fake_client.models.list.return_value = fake_models
        with patch.dict("sys.modules", {
            "openai": MagicMock(OpenAI=lambda api_key: fake_client),
        }):
            r = preflight._check_openai_modelo("sk-" + "x" * 20, "gpt-4o", [])
        assert r.ok
        assert r.detalhes["fallback_em_uso"] is False

    def test_fallback_quando_principal_indisponivel(self):
        fake_models = MagicMock()
        fake_models.data = [type("M", (), {"id": "gpt-4o-mini"})()]
        fake_client = MagicMock()
        fake_client.models.list.return_value = fake_models
        with patch.dict("sys.modules", {
            "openai": MagicMock(OpenAI=lambda api_key: fake_client),
        }):
            r = preflight._check_openai_modelo(
                "sk-" + "x" * 20, "gpt-5.1-chat-latest", ["gpt-4o", "gpt-4o-mini"],
            )
        assert r.ok
        assert r.detalhes["fallback_em_uso"] is True
        assert r.detalhes["fallback"] == "gpt-4o-mini"

    def test_nenhum_modelo_acessivel(self):
        fake_models = MagicMock()
        fake_models.data = [type("M", (), {"id": "gpt-3.5-turbo"})()]
        fake_client = MagicMock()
        fake_client.models.list.return_value = fake_models
        with patch.dict("sys.modules", {
            "openai": MagicMock(OpenAI=lambda api_key: fake_client),
        }):
            r = preflight._check_openai_modelo(
                "sk-" + "x" * 20, "gpt-5.1-chat-latest", ["gpt-4o"],
            )
        assert not r.ok


# ─── executar_preflight com sondar_modelos_llm ──────────────────────────────

class TestSondarModelosFlag:
    def test_default_nao_sonda(self):
        # Sem sondar_modelos_llm, nao deve adicionar 'gemini_modelo'/'openai_modelo'
        with patch("src.core.preflight._check_lm_studio") as mock_lms, \
             patch("src.core.preflight._check_modelo_lstm") as mock_lstm, \
             patch("src.core.preflight._check_contexto_md") as mock_md, \
             patch("src.core.preflight._check_sabesp_ping") as mock_s:
            from src.core.preflight import CheckResult
            mock_lms.return_value = CheckResult("lm_studio", True, "ok")
            mock_lstm.return_value = CheckResult("hydro_lstm", True, "ok")
            mock_md.return_value = CheckResult("contexto_md", True, "ok")
            mock_s.return_value = CheckResult("sabesp_api", True, "ok")
            rs = preflight.executar_preflight(sondar_modelos_llm=False)
        nomes = [r.nome for r in rs]
        assert "gemini_modelo" not in nomes
        assert "openai_modelo" not in nomes

    def test_sondar_true_chama_funcoes(self):
        with patch("src.core.preflight._check_lm_studio") as mock_lms, \
             patch("src.core.preflight._check_modelo_lstm") as mock_lstm, \
             patch("src.core.preflight._check_contexto_md") as mock_md, \
             patch("src.core.preflight._check_sabesp_ping") as mock_s, \
             patch("src.core.preflight._check_gemini_modelo") as mock_g, \
             patch("src.core.preflight._check_openai_modelo") as mock_o:
            from src.core.preflight import CheckResult
            mock_lms.return_value = CheckResult("lm_studio", True, "ok")
            mock_lstm.return_value = CheckResult("hydro_lstm", True, "ok")
            mock_md.return_value = CheckResult("contexto_md", True, "ok")
            mock_s.return_value = CheckResult("sabesp_api", True, "ok")
            mock_g.return_value = CheckResult("gemini_modelo", True, "ok")
            mock_o.return_value = CheckResult("openai_modelo", True, "ok")
            rs = preflight.executar_preflight(sondar_modelos_llm=True)
        nomes = [r.nome for r in rs]
        assert "gemini_modelo" in nomes
        assert "openai_modelo" in nomes
