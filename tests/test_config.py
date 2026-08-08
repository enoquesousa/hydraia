"""Testes de carga e validacao de config_apis.json."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import ConfigError, carregar_config, selecionar_modelo


# ─── carregar_config ─────────────────────────────────────────────────────────

def _escrever(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestCarregarConfig:
    def test_le_config_valida(self, tmp_path):
        p = tmp_path / "c.json"
        _escrever(p, {
            "openweather": {"base_url": "u", "timeout": 10, "retry_attempts": 3},
            "gemini_ultra": {"model": "gemini-2.5-pro", "max_tokens": 2048, "temperature": 0.3},
            "openai_gpt5": {"model": "gpt-4o", "max_tokens": 2048, "temperature": 0.3},
        })
        cfg = carregar_config(p)
        assert cfg["gemini_ultra"]["model"] == "gemini-2.5-pro"
        # fallbacks default sao injetados
        assert isinstance(cfg["gemini_ultra"]["fallback_models"], list)

    def test_falha_se_secao_faltando(self, tmp_path):
        p = tmp_path / "c.json"
        _escrever(p, {"openweather": {}})
        with pytest.raises(ConfigError):
            carregar_config(p)

    def test_falha_se_tipo_errado(self, tmp_path):
        p = tmp_path / "c.json"
        _escrever(p, {
            "openweather": {"base_url": "u", "timeout": "errado", "retry_attempts": 3},
            "gemini_ultra": {"model": "x", "max_tokens": 100, "temperature": 0.5},
            "openai_gpt5": {"model": "y", "max_tokens": 100, "temperature": 0.5},
        })
        with pytest.raises(ConfigError):
            carregar_config(p)

    def test_falha_se_arquivo_nao_existe(self, tmp_path):
        with pytest.raises(ConfigError):
            carregar_config(tmp_path / "nao_existe.json")

    def test_falha_se_json_invalido(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text("{lixo")
        with pytest.raises(ConfigError):
            carregar_config(p)

    def test_env_override_aplica(self, tmp_path, monkeypatch):
        p = tmp_path / "c.json"
        _escrever(p, {
            "openweather": {"base_url": "u", "timeout": 10, "retry_attempts": 3},
            "gemini_ultra": {"model": "gemini-2.5-pro", "max_tokens": 2048, "temperature": 0.3},
            "openai_gpt5": {"model": "gpt-4o", "max_tokens": 2048, "temperature": 0.3},
        })
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
        monkeypatch.setenv("GPT_MODEL", "gpt-4o-mini")
        cfg = carregar_config(p)
        assert cfg["gemini_ultra"]["model"] == "gemini-1.5-pro"
        assert cfg["openai_gpt5"]["model"] == "gpt-4o-mini"


# ─── selecionar_modelo ───────────────────────────────────────────────────────

class TestSelecionarModelo:
    def test_principal_disponivel(self):
        cfg = {"model": "a", "fallback_models": ["b", "c"]}
        out = selecionar_modelo(cfg, sondador=lambda m: m == "a")
        assert out == "a"

    def test_fallback_quando_principal_indisponivel(self):
        cfg = {"model": "a", "fallback_models": ["b", "c"]}
        out = selecionar_modelo(cfg, sondador=lambda m: m == "b")
        assert out == "b"

    def test_pula_para_terceiro_se_dois_indisponiveis(self):
        cfg = {"model": "a", "fallback_models": ["b", "c"]}
        out = selecionar_modelo(cfg, sondador=lambda m: m == "c")
        assert out == "c"

    def test_retorna_none_se_nenhum_disponivel(self):
        cfg = {"model": "a", "fallback_models": ["b"]}
        out = selecionar_modelo(cfg, sondador=lambda m: False)
        assert out is None

    def test_excecao_no_sondador_nao_quebra(self):
        cfg = {"model": "a", "fallback_models": ["b"]}
        def sondar(m):
            if m == "a":
                raise RuntimeError("erro temporario")
            return True
        out = selecionar_modelo(cfg, sondador=sondar)
        assert out == "b"
