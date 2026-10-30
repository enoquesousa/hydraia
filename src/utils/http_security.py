"""Fronteiras HTTP dos clientes que transportam chaves ou prompts."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit


def validar_url_servico(url: str, *, permitir_http_loopback: bool = False) -> str:
    """Exige HTTPS, exceto HTTP local explicitamente permitido pelo cliente.

    A URL nao pode carregar credenciais, query ou fragmento. A mensagem de
    configuracao nunca ecoa o valor recebido, mesmo quando urlsplit falha.
    Nomes de host nao sao resolvidos para decidir se HTTP e permitido.
    """
    mensagem = (
        "URL de servico invalida: use HTTPS sem credenciais, query ou fragmento; "
        "HTTP e permitido apenas para o cliente local em loopback."
    )
    try:
        if not isinstance(url, str) or not url or any(
            c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url
        ):
            raise ValueError
        partes = urlsplit(url)
        host = partes.hostname
        porta = partes.port
        if (
            partes.scheme not in {"http", "https"}
            or not host
            or partes.username is not None
            or partes.password is not None
            or "?" in url
            or "#" in url
            or "\\" in url
            or "%" in host
            or (porta is not None and not 1 <= porta <= 65535)
            or partes.netloc.endswith(":")
        ):
            raise ValueError
        local = host.lower() == "localhost"
        if not local:
            try:
                local = ip_address(host).is_loopback
            except ValueError:
                local = False
        if partes.scheme == "http" and not (permitir_http_loopback and local):
            raise ValueError
    except ValueError:
        raise ValueError(mensagem) from None
    return url.rstrip("/")


def exigir_resposta_sem_redirecionamento(response) -> None:
    """raise_for_status ignora 3xx; nunca interprete um redirect como sucesso."""
    status = response.status_code
    if isinstance(status, int) and 300 <= status < 400:
        raise RuntimeError("Redirecionamento HTTP recusado para este servico.")
