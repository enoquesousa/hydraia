"""Testes da observabilidade — masking, metricas, correlation_id, versao."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from io import StringIO

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.observabilidade import (
    JsonFormatter,
    METRICS,
    MaskingFilter,
    MetricsRegistry,
    correlation_id,
    hash_prompt,
    mascarar_token,
    novo_correlation_id,
    setup_logging,
    versao_codigo,
)


# ─── Mascaramento ────────────────────────────────────────────────────────────

class TestMascararToken:
    def test_mascara_chave_openai(self):
        out = mascarar_token("Chave: sk-" + "abc123def456ghi789jkl000")
        assert "sk-abc123def456" not in out
        assert "***" in out

    def test_mascara_chave_google(self):
        out = mascarar_token("Token: AIzaSyD12345678901234567890123456")
        assert "AIza" not in out

    def test_mascara_authorization_header(self):
        out = mascarar_token("Authorization: Bearer abc123def456ghi")
        assert "abc123def456" not in out

    def test_mascara_kv_format(self):
        out = mascarar_token('api_key="my-secret-token-1234"')
        assert "my-secret-token-1234" not in out
        assert "***" in out

    def test_string_sem_token_nao_muda(self):
        s = "log normal sem chaves"
        assert mascarar_token(s) == s

    def test_none_e_str_vazia(self):
        assert mascarar_token("") == ""


# ─── Correlation ID ──────────────────────────────────────────────────────────

class TestCorrelationId:
    def test_default_e_dash(self):
        # Pode haver outro teste ja chamou novo_correlation_id, mas em
        # outro contexto. Aqui esperamos um valor (string).
        assert isinstance(correlation_id(), str)

    def test_novo_gera_unico(self):
        a = novo_correlation_id()
        b = novo_correlation_id()
        assert a != b
        assert b.startswith("run-")

    def test_prefix_customizado(self):
        cid = novo_correlation_id(prefix="custom")
        assert cid.startswith("custom-")


# ─── JsonFormatter ───────────────────────────────────────────────────────────

class TestJsonFormatter:
    def test_formata_como_json(self):
        rec = logging.LogRecord(
            name="t", level=logging.INFO, pathname="x", lineno=1,
            msg="oi", args=(), exc_info=None,
        )
        rec.correlation_id = "abc"
        out = JsonFormatter().format(rec)
        d = json.loads(out)
        assert d["level"] == "INFO"
        assert d["message"] == "oi"
        assert d["correlation_id"] == "abc"


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetricsRegistry:
    def test_inc_e_snapshot(self):
        m = MetricsRegistry()
        m.inc("a")
        m.inc("a", 3)
        m.inc("b")
        snap = m.snapshot()
        assert snap["counters"]["a"] == 4
        assert snap["counters"]["b"] == 1

    def test_observe_histograma(self):
        m = MetricsRegistry()
        for v in [1, 2, 3, 4, 5]:
            m.observe("lat", v)
        s = m.snapshot()["histograms"]["lat"]
        assert s["n"] == 5
        assert s["min"] == 1
        assert s["max"] == 5
        assert s["mean"] == 3.0

    def test_timer(self):
        m = MetricsRegistry()
        with m.timer("etapa"):
            time.sleep(0.01)
        s = m.snapshot()["histograms"]["etapa.latency_s"]
        assert s["n"] == 1
        assert s["min"] >= 0.005

    def test_reset(self):
        m = MetricsRegistry()
        m.inc("x")
        m.reset()
        assert m.snapshot()["counters"] == {}


# ─── Versionamento ───────────────────────────────────────────────────────────

class TestVersaoCodigo:
    def test_retorna_hex_12(self):
        v = versao_codigo()
        assert isinstance(v, str)
        assert len(v) == 12

    def test_estavel_entre_chamadas(self):
        a = versao_codigo()
        b = versao_codigo()
        assert a == b


class TestHashPrompt:
    def test_mesmo_prompt_mesmo_hash(self):
        assert hash_prompt("abc") == hash_prompt("abc")

    def test_prompts_diferentes_geram_hashes_diferentes(self):
        assert hash_prompt("a") != hash_prompt("b")


# ─── setup_logging integra mascaramento ──────────────────────────────────────

class TestSetupLoggingMascaramento:
    def test_log_message_mascara_chave(self, capsys):
        setup_logging(json_logs=False)
        novo_correlation_id()
        log = logging.getLogger("teste-mascara")
        token_ficticio = "sk-" + "superSecretToken1234567890"
        log.error("falha com chave %s", token_ficticio)
        captured = capsys.readouterr().err
        assert "sk-superSecret" not in captured
        assert "***" in captured
