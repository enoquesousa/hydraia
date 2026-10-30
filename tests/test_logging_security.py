"""Regressoes de credenciais nos handlers de logging instalados."""

from __future__ import annotations

import json
import logging
import sys
from urllib.parse import parse_qsl, quote, urlsplit

import pytest
import requests

from src.utils.observabilidade import JsonFormatter, mascarar_token, setup_logging


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.parametrize(
    "query",
    [
        "appid={plain}&appid={encoded}&units=metric",
        "%61ppid={encoded}&APPID={plain}&empty=",
        "key={encoded}&access_token={plain}",
        "custom_provider_credential={encoded}&custom_provider_credential={plain}",
    ],
)
def test_url_query_values_are_redacted_including_duplicates_and_encoding(query):
    secret = "synthetic/query+credential&not-a-real-key"
    encoded = quote(secret, safe="")
    # A bare value is independently valid as a query value.
    plain = "syntheticPlainQueryCredential"
    raw_query = query.format(plain=plain, encoded=encoded)
    result = mascarar_token("https://api.example.test/forecast?" + raw_query)
    parsed = urlsplit(result)
    original_pairs = parse_qsl(raw_query, keep_blank_values=True)
    safe_pairs = parse_qsl(parsed.query, keep_blank_values=True)

    assert secret not in result and encoded not in result and plain not in result
    assert parsed.netloc == "api.example.test"
    assert parsed.path == "/forecast"
    assert [key for key, _ in safe_pairs] == [key for key, _ in original_pairs]
    assert all(value == "***" for _, value in safe_pairs)


@pytest.mark.parametrize(
    "message",
    [
        "appid={secret}",
        "%61ppid={secret}&units=metric",
        "OPENWEATHER_API_KEY='{secret}'",
        "Authorization: Bearer {secret}",
        '{{"Authorization": "Bearer {secret}"}}',
        "{{'X-API-Key': '{secret}'}}",
        "Proxy-Authorization: Basic {secret}",
        "access_token={secret}; refresh_token={secret}",
    ],
)
def test_credential_fields_and_headers_are_redacted(message):
    secret = "synthetic.header+credential/with_padding=="
    result = mascarar_token(message.format(secret=secret))
    assert secret not in result
    assert "***" in result


def test_short_values_userinfo_and_fragment_are_not_exempt_from_redaction():
    result = mascarar_token("appid=x https://user:short@api.example.test/p?key=y#private")
    assert "appid=x" not in result
    assert "user:short" not in result
    assert "private" not in result
    assert "key=y" not in result


def _raise_http_error():
    prepared = requests.Request(
        "GET",
        "https://api.example.test/onecall",
        params=[("appid", "synthetic_query_value"), ("appid", "second_query_value")],
    ).prepare()
    response = requests.Response()
    response.status_code = 401
    response.url = prepared.url
    response.request = prepared
    response.reason = "SYNTHETIC_BARE_REASON"
    response.raise_for_status()


def _log_exception_chain(log):
    try:
        _raise_http_error()
    except requests.HTTPError as original:
        original.add_note("SYNTHETIC_PRIVATE_NOTE")
        try:
            raise RuntimeError("SYNTHETIC_BARE_WRAPPER") from original
        except RuntimeError:
            log.exception("consulta falhou", stack_info=True)  # SYNTHETIC_SOURCE_COMMENT


@pytest.mark.parametrize("json_logs", [False, True], ids=["text", "json"])
def test_installed_console_and_file_omit_exception_values_but_keep_frames(
    json_logs, tmp_path, capsys
):
    path = tmp_path / "local-only-log.jsonl"
    setup_logging(json_logs=json_logs, arquivo=path)
    _log_exception_chain(logging.getLogger("hydraia.security.test"))

    console = capsys.readouterr().err
    saved = path.read_text(encoding="utf-8")
    for output in (console, saved):
        for secret in (
            "synthetic_query_value",
            "second_query_value",
            "SYNTHETIC_BARE_REASON",
            "SYNTHETIC_PRIVATE_NOTE",
            "SYNTHETIC_BARE_WRAPPER",
            "SYNTHETIC_SOURCE_COMMENT",
        ):
            assert secret not in output
        assert "consulta falhou" in output
        assert "HTTPError" in output and "RuntimeError" in output
        assert "_raise_http_error" in output and "_log_exception_chain" in output

    file_record = json.loads(saved)
    assert "exc_info" in file_record and "stack_info" in file_record
    assert "_log_exception_chain" in file_record["stack_info"]
    if json_logs:
        assert json.loads(console)["exc_info"] == file_record["exc_info"]


@pytest.mark.parametrize("json_logs", [False, True])
def test_cached_exception_text_is_not_reused_by_installed_formatters(json_logs, capsys):
    setup_logging(json_logs=json_logs)
    try:
        raise ValueError("SYNTHETIC_EXCEPTION_VALUE")
    except ValueError:
        record = logging.LogRecord(
            "hydraia.security.test",
            logging.ERROR,
            __file__,
            1,
            "erro seguro",
            (),
            sys.exc_info(),
        )
    record.exc_text = "SYNTHETIC_CACHED_EXCEPTION"
    logging.getLogger(record.name).handle(record)
    output = capsys.readouterr().err
    assert "SYNTHETIC_CACHED_EXCEPTION" not in output
    assert "SYNTHETIC_EXCEPTION_VALUE" not in output
    assert "ValueError" in output and "erro seguro" in output


def test_json_formatter_does_not_call_exception_string_or_include_exception_group_values():
    class SensitiveError(Exception):
        def __str__(self):
            raise AssertionError("Exception values must not be formatted")

    try:
        raise ExceptionGroup("SYNTHETIC_GROUP_VALUE", [SensitiveError()])
    except ExceptionGroup:
        record = logging.LogRecord("test", logging.ERROR, __file__, 1, "falha", (), sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ExceptionGroup" in payload["exc_info"]
    assert "SensitiveError" in payload["exc_info"]
    assert "SYNTHETIC_GROUP_VALUE" not in payload["exc_info"]


@pytest.mark.parametrize("json_logs", [False, True])
def test_installed_handlers_sanitize_messages_and_fail_closed_on_bad_interpolation(
    json_logs, tmp_path, capsys
):
    path = tmp_path / "local-only-messages.jsonl"
    setup_logging(json_logs=json_logs, arquivo=path)
    log = logging.getLogger("hydraia.security.test")
    log.error(
        "URL https://api.example.test/p?%%61ppid=%s&key=%s",
        "synthetic-encoded-field",
        "synthetic-second-field",
    )
    log.error("valor inteiro: %d", "SYNTHETIC_UNFORMATTABLE_ARG")
    for output in (capsys.readouterr().err, path.read_text(encoding="utf-8")):
        assert "synthetic-encoded-field" not in output
        assert "synthetic-second-field" not in output
        assert "SYNTHETIC_UNFORMATTABLE_ARG" not in output
        assert "Mensagem de log indisponivel." in output
        assert "api.example.test/p" in output


def test_normal_diagnostics_are_preserved():
    message = "OpenWeather: timeout apos 10 segundos na tentativa 2/3"
    assert mascarar_token(message) == message
