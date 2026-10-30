"""Collector com requests real preparado, sem envio de chamadas externas."""

from __future__ import annotations

import io
import logging

import pytest
import requests

from src.api.openweather import OpenWeatherCollector


@pytest.fixture(autouse=True)
def synthetic_environment(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "synthetic-collector-api-credential")

    def forbid_send(*args, **kwargs):
        raise AssertionError("A regression test must never send a request")

    monkeypatch.setattr(requests.sessions.Session, "send", forbid_send)


@pytest.fixture
def unfiltered_log():
    log = logging.getLogger("src.api.openweather")
    handlers, propagate, level = list(log.handlers), log.propagate, log.level
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG)
    yield output
    log.handlers[:] = handlers
    log.propagate = propagate
    log.setLevel(level)
    handler.close()


def _response(status, endpoint, params, reason="Unauthorized"):
    prepared = requests.Request("GET", endpoint, params=params).prepare()
    response = requests.Response()
    response.request = prepared
    response.url = prepared.url
    response.status_code = status
    response.reason = reason
    return response


def test_http_error_does_not_log_url_response_body_or_exception_text(monkeypatch, unfiltered_log):
    calls = []

    def failed_get(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        response = _response(401, endpoint, kwargs["params"], "SYNTHETIC_RAW_REASON")
        response._content = b"SYNTHETIC_RAW_BODY"
        return response

    monkeypatch.setattr(requests, "get", failed_get)
    collector = OpenWeatherCollector({"retry_attempts": 1})
    assert collector.coletar_previsao(-23.5, -46.6) is None
    output = unfiltered_log.getvalue()
    for secret in (collector.api_key, "SYNTHETIC_RAW_REASON", "SYNTHETIC_RAW_BODY", "appid="):
        assert secret not in output
    assert "HTTPError" in output and "401" in output
    assert len(calls) == 1 and calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["params"]["appid"] == collector.api_key


def test_request_exception_keeps_retries_without_formatting_arbitrary_values(
    monkeypatch, unfiltered_log
):
    attempts = []
    sleeps = []

    class OpaqueRequestError(requests.RequestException):
        def __str__(self):
            raise AssertionError("The collector must not format the exception")

    def failed_get(*args, **kwargs):
        attempts.append(kwargs)
        raise OpaqueRequestError("SYNTHETIC_UNSTRUCTURED_CREDENTIAL")

    monkeypatch.setattr(requests, "get", failed_get)
    monkeypatch.setattr("src.api.openweather.time.sleep", sleeps.append)
    collector = OpenWeatherCollector({"retry_attempts": 2})
    assert collector.coletar_previsao(0, 0) is None
    assert len(attempts) == 2 and sleeps == [1]
    assert "RequestException" in unfiltered_log.getvalue()
    assert "SYNTHETIC_UNSTRUCTURED_CREDENTIAL" not in unfiltered_log.getvalue()


@pytest.mark.parametrize("status", [True, 0, "SYNTHETIC_INVALID_STATUS"])
def test_only_valid_integer_http_status_is_logged(monkeypatch, unfiltered_log, status):
    response = requests.Response()
    response.status_code = status

    def failed_get(*args, **kwargs):
        raise requests.HTTPError("SYNTHETIC_RAW_ERROR", response=response)

    monkeypatch.setattr(requests, "get", failed_get)
    collector = OpenWeatherCollector({"retry_attempts": 1})
    assert collector.coletar_previsao(0, 0) is None
    assert "HTTP indisponivel" in unfiltered_log.getvalue()
    assert "SYNTHETIC_" not in unfiltered_log.getvalue()


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirect_is_refused_before_body_parsing(monkeypatch, unfiltered_log, status):
    calls = []

    def redirect_get(endpoint, **kwargs):
        calls.append(kwargs)
        response = _response(status, endpoint, kwargs["params"])
        response.headers["Location"] = "https://other.example.test/?key=synthetic-redirect-key"

        def no_json():
            raise AssertionError("Redirect bodies are never parsed")

        response.json = no_json
        return response

    monkeypatch.setattr(requests, "get", redirect_get)
    collector = OpenWeatherCollector({"retry_attempts": 3})
    assert collector.coletar_previsao(0, 0) is None
    assert len(calls) == 1 and calls[0]["allow_redirects"] is False
    assert "synthetic-redirect-key" not in unfiltered_log.getvalue()
    assert "redirecionamento" in unfiltered_log.getvalue().lower()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.test/data/3.0",
        "http://127.0.0.1/data/3.0",
        "https://user:synthetic-password@api.example.test/data/3.0",
        "https://api.example.test/data/3.0?key=synthetic-query-key",
        "https://api.example.test/data/3.0#synthetic-fragment",
    ],
)
def test_insecure_or_credential_bearing_base_url_is_rejected_before_request(base_url):
    with pytest.raises(ValueError) as caught:
        OpenWeatherCollector({"base_url": base_url})
    assert base_url not in str(caught.value)
    assert "synthetic-" not in str(caught.value)


def test_successful_response_retains_scientific_data_and_request_options(monkeypatch):
    calls = []

    def successful_get(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        response = _response(200, endpoint, kwargs["params"], "OK")
        response.json = lambda: {
            "lat": -23.5,
            "lon": -46.6,
            "daily": [{"dt": 1700000000, "temp": {"day": 24.5}, "rain": 4.25}],
        }
        return response

    monkeypatch.setattr(requests, "get", successful_get)
    collector = OpenWeatherCollector(
        {
            "base_url": "https://api.example.test/data/3.0/",
            "timeout": 12,
        }
    )
    result = collector.coletar_previsao(-23.5, -46.6)
    assert result["total_previsoes"] == 1
    assert result["previsoes"][0]["temperatura_c"] == 24.5
    assert result["previsoes"][0]["precipitacao_mm"] == 4.25
    assert result["coordenadas"] == {"lat": -23.5, "lon": -46.6}
    endpoint, options = calls[0]
    assert endpoint == "https://api.example.test/data/3.0/onecall"
    assert options["timeout"] == 12 and options["allow_redirects"] is False
    assert options["params"]["units"] == "metric"
