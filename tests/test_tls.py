"""Testes da configuracao TLS nativa."""

from unittest.mock import MagicMock

from src.utils import tls


def test_configurar_certificados_nativos(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "truststore", fake)
    monkeypatch.delenv("HYDRAIA_NATIVE_CA", raising=False)
    monkeypatch.setattr(tls, "_CONFIGURADO", False)

    assert tls.configurar_certificados_nativos() is True
    fake.inject_into_ssl.assert_called_once_with()


def test_configuracao_pode_ser_desabilitada(monkeypatch):
    monkeypatch.setenv("HYDRAIA_NATIVE_CA", "0")
    monkeypatch.setattr(tls, "_CONFIGURADO", False)

    assert tls.configurar_certificados_nativos() is False
