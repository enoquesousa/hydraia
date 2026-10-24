"""Regressoes de transporte e diagnosticos sem chaves, com HTTP simulado."""

from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import pytest
import requests

from src.core import preflight
from src.utils import listar_modelos_llm
from src.utils.env_loader import EnvLoader
from src.utils.paths import HydraPaths


SECRET = "synthetic-credential-marker"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Nunca consulta APIs ou um .env fora do diretorio temporario do teste."""
    monkeypatch.setattr(HydraPaths, "BASE", tmp_path)
    for name in (
        "LM_STUDIO_URL",
        "OPENWEATHER_API_KEY",
        "OPENAI_GPT5_API_KEY",
        "GEMINI_ULTRA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        Mock(side_effect=AssertionError("Rede externa proibida neste teste")),
    )


def response(status=200, payload=None):
    result = requests.Response()
    result.status_code = status
    result.url = f"https://provider.invalid/models?key={SECRET}"
    result.headers["Location"] = f"https://redirect.invalid/{SECRET}"
    if payload is None:
        payload = {"data": [{"id": "chat-model"}], "succeeded": True}
    result._content = json.dumps(payload).encode()
    return result


def assert_safe_report(result, capsys):
    preflight.imprimir_resumo([result])
    serialized = json.dumps(preflight.relatorio_dict([result]))
    output = capsys.readouterr()
    assert SECRET not in serialized
    assert SECRET not in output.out
    assert SECRET not in output.err


def test_openweather_preflight_uses_https_params_and_no_redirects(monkeypatch):
    get = Mock(return_value=response())
    monkeypatch.setattr(requests, "get", get)

    result = preflight._check_openweather_ping(SECRET, timeout_s=3.5)

    assert result.ok
    assert not result.obrigatorio
    get.assert_called_once_with(
        "https://api.openweathermap.org/geo/1.0/reverse",
        params={"lat": -23.5505, "lon": -46.6333, "limit": 1, "appid": SECRET},
        timeout=3.5,
        allow_redirects=False,
    )


@pytest.mark.parametrize(
    "probe,args",
    [
        (preflight._check_openweather_ping, (SECRET,)),
        (preflight._check_lm_studio, ("http://localhost:1234/v1",)),
        (preflight._check_sabesp_ping, ()),
    ],
    ids=["openweather", "lm-studio", "sabesp"],
)
@pytest.mark.parametrize("failure", ["network", "http", "redirect"])
def test_preflight_errors_do_not_expose_request_or_response(
    probe, args, failure, monkeypatch, capsys
):
    if failure == "network":
        get = Mock(side_effect=requests.exceptions.ConnectionError(f"GET key={SECRET}"))
    else:
        # O erro HTTP contem a URL; o redirect contem Location e JSON plausivel.
        get = Mock(return_value=response(401 if failure == "http" else 302))
    monkeypatch.setattr(requests, "get", get)

    result = probe(*args)

    assert not result.ok
    assert get.call_count == 1
    assert get.call_args.kwargs["allow_redirects"] is False
    assert_safe_report(result, capsys)


@pytest.mark.parametrize(
    "url",
    [
        f"http://{SECRET}.invalid/v1",
        f"https://user:{SECRET}@provider.invalid/v1",
        f"https://provider.invalid/v1?key={SECRET}",
        f"https://provider.invalid:{SECRET}/v1",
        f"https://provider.invalid/v1#{SECRET}",
    ],
)
def test_lm_studio_invalid_url_is_not_requested_or_reported(url, monkeypatch, capsys):
    get = Mock()
    monkeypatch.setattr(requests, "get", get)

    result = preflight._check_lm_studio(url)

    assert not result.ok
    assert result.obrigatorio
    assert result.detalhes == {}
    get.assert_not_called()
    assert_safe_report(result, capsys)


@pytest.mark.parametrize("probe", [preflight._check_lm_studio, preflight._check_sabesp_ping])
def test_preflight_invalid_json_becomes_safe_failure(probe, monkeypatch, capsys):
    upstream = response()
    upstream._content = SECRET.encode()
    monkeypatch.setattr(requests, "get", Mock(return_value=upstream))

    result = probe()

    assert not result.ok
    assert_safe_report(result, capsys)


def test_model_probe_setup_failure_is_not_serialized(monkeypatch, capsys):
    for name in (
        "_check_chave",
        "_check_lm_studio",
        "_check_modelo_lstm",
        "_check_contexto_md",
        "_check_sabesp_ping",
    ):
        monkeypatch.setattr(preflight, name, Mock(return_value=preflight.CheckResult(name, True)))
    with patch("src.core.config.carregar_config", side_effect=ValueError(SECRET)):
        results = preflight.executar_preflight(sondar_modelos_llm=True)

    result = next(item for item in results if item.nome == "modelos_llm_sondagem")
    assert not result.ok
    assert not result.obrigatorio
    assert_safe_report(result, capsys)


@pytest.mark.parametrize(
    "listing,env_name",
    [
        (listar_modelos_llm.listar_modelos_openai, "OPENAI_GPT5_API_KEY"),
        (listar_modelos_llm.listar_modelos_gemini, "GEMINI_ULTRA_API_KEY"),
    ],
    ids=["openai", "gemini"],
)
@pytest.mark.parametrize("failure", ["network", "http", "redirect", "json"])
def test_model_listing_errors_hide_credentials_and_body(
    listing, env_name, failure, monkeypatch, capsys
):
    monkeypatch.setenv(env_name, SECRET)
    if failure == "network":
        get = Mock(side_effect=requests.exceptions.ConnectionError(f"GET key={SECRET}"))
    else:
        upstream = response(
            {"http": 401, "redirect": 302, "json": 200}[failure],
            {"data": [{"id": SECRET}], "models": [{"name": SECRET}]},
        )
        if failure == "json":
            upstream._content = SECRET.encode()
        get = Mock(return_value=upstream)
    monkeypatch.setattr(requests, "get", get)

    assert listing() == []

    assert get.call_count == 1
    assert get.call_args.kwargs["allow_redirects"] is False
    output = capsys.readouterr()
    assert "Erro ao listar modelos" in output.out
    assert SECRET not in output.out
    assert SECRET not in output.err


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_model_listing_keeps_valid_result_and_transport_contract(provider, monkeypatch):
    if provider == "openai":
        listing = listar_modelos_llm.listar_modelos_openai
        env_name, field, model_key = "OPENAI_GPT5_API_KEY", "data", "id"
        url = "https://api.openai.com/v1/models"
        auth = {"headers": {"Authorization": f"Bearer {SECRET}"}}
    else:
        listing = listar_modelos_llm.listar_modelos_gemini
        env_name, field, model_key = "GEMINI_ULTRA_API_KEY", "models", "name"
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        auth = {"params": {"key": SECRET}}
    monkeypatch.setenv(env_name, SECRET)
    get = Mock(return_value=response(payload={field: [{model_key: "z"}, {model_key: "a"}]}))
    monkeypatch.setattr(requests, "get", get)

    assert listing() == [{model_key: "a"}, {model_key: "z"}]
    get.assert_called_once_with(url, **auth, timeout=15, allow_redirects=False)


@pytest.mark.parametrize("getter,default", [(EnvLoader.get_int, 7), (EnvLoader.get_float, 2.5)])
def test_invalid_numeric_setting_keeps_default_without_printing_value(
    getter, default, monkeypatch, capsys
):
    monkeypatch.setenv("HYDRAIA_TEST_NUMBER", SECRET)

    assert getter("HYDRAIA_TEST_NUMBER", default) == default

    assert SECRET not in capsys.readouterr().out


def test_dotenv_read_failure_does_not_print_exception(tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch("builtins.open", side_effect=OSError(SECRET)):
        EnvLoader(str(env_file))

    output = capsys.readouterr().out
    assert "Erro ao carregar .env" in output
    assert SECRET not in output


def test_explicit_dotenv_preserves_environment_and_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING=file-value\nNEW=value=with-equals\n", encoding="utf-8")
    with patch.dict(os.environ, {"EXISTING": "environment-value"}, clear=True):
        EnvLoader(str(env_file))
        assert os.environ["EXISTING"] == "environment-value"
        assert os.environ["NEW"] == "value=with-equals"


@pytest.mark.parametrize(
    "env_name,placeholder",
    [
        ("OPENWEATHER_API_KEY", "your_openweather_key_here"),
        ("GEMINI_ULTRA_API_KEY", "your_gemini_key_here"),
        ("OPENAI_GPT5_API_KEY", "your_openai_key_here"),
    ],
)
def test_template_keys_are_not_reported_as_configured(env_name, placeholder, monkeypatch):
    value = f"  {placeholder.upper()}  "
    monkeypatch.setenv(env_name, value)

    assert not preflight._api_key_valida(value, env_name)
    assert EnvLoader.validar_chaves_obrigatorias([env_name]) == {env_name: False}

    # Este check local nao tenta autenticar nem invalidar chaves no provedor.
    monkeypatch.setenv(env_name, SECRET)
    assert preflight._api_key_valida(SECRET, env_name)
    assert EnvLoader.validar_chaves_obrigatorias([env_name]) == {env_name: True}
