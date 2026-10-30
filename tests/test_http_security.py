"""Regressoes de transporte e de erros, sem chamar provedores reais."""

import json
import logging
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest
import requests

from src.core.llm_specialists import GPTSpecialist, LLMSpecialist, StopRetry
from src.models import hydraia_llm
from src.utils.http_security import validar_url_servico


@pytest.fixture(autouse=True)
def ambiente_isolado(monkeypatch):
    for nome in (
        "LM_STUDIO_URL", "LM_STUDIO_NATIVE_CHAT_URL", "LM_STUDIO_MODEL",
        "LM_STUDIO_TIMEOUT", "LM_STUDIO_REASONING",
    ):
        monkeypatch.delenv(nome, raising=False)
    hydraia_llm._modelo_cache = None
    yield
    hydraia_llm._modelo_cache = None


@pytest.mark.parametrize("url", [
    "http://api.example.test/v1", "http://192.168.1.2/v1", "http://0.0.0.0/v1",
    "http://localhost.example.test/v1", "http://2130706433/v1",
    "https://user:synthetic-value@example.test/v1", "https://example.test/v1?key=x",
    "https://example.test/v1?", "https://example.test/v1#", "https://example.test:0/v1",
    "https://example.test:65536/v1", "https://example.test:/v1", "https:///v1",
    "https://example.test\\@localhost/v1", "\nhttps://example.test/v1",
    "https://example.test/\tpath", "https://[::1", "https://%65xample.test/v1",
])
def test_destino_invalido_recusado_antes_de_descobrir_ou_enviar(url, monkeypatch):
    monkeypatch.setenv("LM_STUDIO_URL", url)
    get = Mock()
    post = Mock()
    monkeypatch.setattr(hydraia_llm.requests, "get", get)
    monkeypatch.setattr(hydraia_llm.requests, "post", post)
    with pytest.raises(ValueError, match="URL de servico invalida") as erro:
        hydraia_llm.ask_local_llm("prompt sintetico")
    assert url not in str(erro.value)
    get.assert_not_called()
    post.assert_not_called()


def test_override_nativo_tambem_validado_antes_da_descoberta(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_NATIVE_CHAT_URL", "http://203.0.113.7/api/v1/chat")
    get = Mock()
    post = Mock()
    monkeypatch.setattr(hydraia_llm.requests, "get", get)
    monkeypatch.setattr(hydraia_llm.requests, "post", post)
    with pytest.raises(ValueError):
        hydraia_llm.ask_local_llm("prompt sintetico")
    get.assert_not_called()
    post.assert_not_called()


@pytest.mark.parametrize("url", [
    "http://localhost:1234/v1", "http://127.0.0.1:1234/v1",
    "http://[::1]:1234/v1", "https://llm.example.test/v1/",
])
def test_local_explicito_ou_https_preserva_destino(url):
    assert validar_url_servico(url, permitir_http_loopback=True) == url.rstrip("/")


def test_cliente_de_chaves_exige_https_inclusive_em_loopback():
    with pytest.raises(ValueError):
        validar_url_servico("http://127.0.0.1:1234/v1")
    assert validar_url_servico("https://api.example.test/v1") == "https://api.example.test/v1"


def resposta(status, payload=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://llm.example.test/api/v1/chat"
    result._content = json.dumps(payload or {}).encode()
    return result


@pytest.mark.parametrize("primeiro_status", [None, 400, 404])
def test_redirect_recusado_no_nativo_retry_e_fallback(primeiro_status, monkeypatch):
    monkeypatch.setenv("LM_STUDIO_MODEL", "offline-test")
    redirect = resposta(307)
    redirect.headers["Location"] = "http://203.0.113.7/capture"
    respostas = [redirect] if primeiro_status is None else [resposta(primeiro_status), redirect]
    post = Mock(side_effect=respostas)
    monkeypatch.setattr(hydraia_llm.requests, "post", post)
    with pytest.raises(RuntimeError, match="Redirecionamento HTTP recusado"):
        hydraia_llm.ask_local_llm("prompt sintetico")
    assert post.call_count == len(respostas)
    assert all(call.kwargs["allow_redirects"] is False for call in post.call_args_list)


def test_redirect_recusado_na_descoberta(monkeypatch):
    get = Mock(return_value=resposta(302))
    monkeypatch.setattr(hydraia_llm.requests, "get", get)
    with pytest.raises(RuntimeError, match="Redirecionamento HTTP recusado"):
        hydraia_llm._descobrir_modelo()
    assert get.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("classe", [
    requests.ConnectionError, requests.Timeout, requests.HTTPError,
])
def test_erro_http_nao_expoe_mensagem_ou_cadeia(classe, monkeypatch):
    marcador = "synthetic-private-provider-payload"
    monkeypatch.setenv("LM_STUDIO_MODEL", "offline-test")
    monkeypatch.setattr(hydraia_llm.requests, "post", Mock(side_effect=classe(marcador)))
    with pytest.raises((ConnectionError, RuntimeError)) as erro:
        hydraia_llm.ask_local_llm("prompt sintetico")
    assert marcador not in "".join(traceback.format_exception(erro.value))


def test_cliente_real_nao_reenvia_prompt_apos_redirect(monkeypatch):
    recebidos = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            recebidos.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(307)
            self.send_header("Location", "/capture")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("LM_STUDIO_MODEL", "offline-test")
    monkeypatch.setenv("LM_STUDIO_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        with pytest.raises(RuntimeError, match="Redirecionamento HTTP recusado"):
            hydraia_llm.ask_local_llm("prompt sintetico sem dados reais")
        assert recebidos == ["/api/v1/chat"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.parametrize("classe", [StopRetry, RuntimeError])
def test_retry_llm_nao_registra_payload_do_provedor(classe, caplog):
    marcador = "synthetic-unlabelled-credential"

    class Specialist(LLMSpecialist):
        def _invocar(self, prompt):
            raise classe(marcador)

    client = Specialist(api_key="synthetic-key", model="offline-test", max_retries=0)
    with caplog.at_level(logging.WARNING):
        assert client._invocar_com_retry("prompt sintetico") is None
    assert marcador not in caplog.text
    assert classe.__name__ in caplog.text


def test_erro_permanente_openai_nao_carrega_payload(monkeypatch):
    import httpx
    import openai

    marcador = "synthetic-private-openai-error"
    http_response = httpx.Response(401, request=httpx.Request("POST", "https://api.example.test"))
    erro = openai.AuthenticationError(marcador, response=http_response, body={"error": marcador})
    fake = Mock()
    fake.responses.create.side_effect = erro
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=fake))
    client = GPTSpecialist(api_key="synthetic-key", model="offline-test")
    with pytest.raises(StopRetry) as resultado:
        client._invocar("prompt sintetico")
    assert marcador not in "".join(traceback.format_exception(resultado.value))


def test_estatistica_invalida_nao_expoe_valor_da_resposta(monkeypatch):
    marcador = "synthetic-private-output-count"
    monkeypatch.setenv("LM_STUDIO_MODEL", "offline-test")
    result = resposta(200, {
        "stats": {"total_output_tokens": marcador},
        "output": [{"type": "message", "content": "resultado"}],
    })
    monkeypatch.setattr(hydraia_llm.requests, "post", Mock(return_value=result))
    with pytest.raises(RuntimeError, match="Resposta de geracao do LM Studio invalida") as erro:
        hydraia_llm.ask_local_llm("prompt sintetico")
    assert marcador not in "".join(traceback.format_exception(erro.value))
