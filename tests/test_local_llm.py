"""Testes offline do cliente OpenAI-compatible do LM Studio."""

from unittest.mock import Mock

import pytest

from src.models import hydraia_llm


@pytest.fixture(autouse=True)
def _limpar_modelo_cache(monkeypatch):
    hydraia_llm._modelo_cache = None
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
    monkeypatch.delenv("LM_STUDIO_TIMEOUT", raising=False)
    monkeypatch.delenv("LM_STUDIO_REASONING", raising=False)
    yield
    hydraia_llm._modelo_cache = None


def test_descobre_chat_e_ignora_embedding(monkeypatch):
    resposta = Mock()
    resposta.raise_for_status.return_value = None
    resposta.json.return_value = {
        "data": [
            {"id": "text-embedding-nomic-embed-text-v1.5"},
            {"id": "hydraia-synth"},
        ]
    }
    get = Mock(return_value=resposta)
    monkeypatch.setattr(hydraia_llm.requests, "get", get)

    assert hydraia_llm._descobrir_modelo() == "hydraia-synth"
    get.assert_called_once()


def test_modelo_explicito_nao_consulta_servidor(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_MODEL", "hydraia-synth")
    get = Mock()
    monkeypatch.setattr(hydraia_llm.requests, "get", get)

    assert hydraia_llm._descobrir_modelo() == "hydraia-synth"
    get.assert_not_called()


def test_timeout_configuravel(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_TIMEOUT", "42.5")
    assert hydraia_llm._timeout_s() == 42.5

    monkeypatch.setenv("LM_STUDIO_TIMEOUT", "invalido")
    assert hydraia_llm._timeout_s() == hydraia_llm.DEFAULT_TIMEOUT_S


def test_chamada_nativa_lm_studio(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_MODEL", "hydraia-synth")
    monkeypatch.setenv("LM_STUDIO_TIMEOUT", "45")
    resposta = Mock()
    resposta.status_code = 200
    resposta.raise_for_status.return_value = None
    resposta.json.return_value = {
        "output": [{"type": "message", "content": "  relatório técnico  "}]
    }
    post = Mock(return_value=resposta)
    monkeypatch.setattr(hydraia_llm.requests, "post", post)

    saida = hydraia_llm.ask_local_llm("sintetize", max_tokens=200, temperature=0.2)

    assert saida == "relatório técnico"
    chamada = post.call_args
    assert chamada.kwargs["timeout"] == 45
    assert chamada.kwargs["json"]["model"] == "hydraia-synth"
    assert chamada.kwargs["json"]["max_output_tokens"] == 200
    assert chamada.kwargs["json"]["reasoning"] == "off"


def test_fallback_para_endpoint_openai_legado(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_MODEL", "hydraia-synth")
    nativa = Mock(status_code=404)
    legada = Mock(status_code=200)
    legada.raise_for_status.return_value = None
    legada.json.return_value = {
        "choices": [{"message": {"content": "fallback funcionando"}}]
    }
    post = Mock(side_effect=[nativa, legada])
    monkeypatch.setattr(hydraia_llm.requests, "post", post)

    assert hydraia_llm.ask_local_llm("teste") == "fallback funcionando"
    assert post.call_count == 2
    assert post.call_args_list[1].args[0].endswith("/v1/chat/completions")
