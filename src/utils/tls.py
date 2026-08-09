"""Configuracao TLS da aplicacao.

Em ambientes corporativos e em maquinas com inspecao HTTPS, a cadeia de
certificados confiada pelo Windows pode diferir do bundle incluido no Python.
O ``truststore`` permite que os clientes HTTP usem o repositorio nativo sem
reduzir a seguranca ou desativar a verificacao TLS.
"""

from __future__ import annotations

import os

_CONFIGURADO = False


def configurar_certificados_nativos() -> bool:
    """Ativa o repositorio de CAs do sistema para os clientes HTTPS.

    Retorna ``True`` quando a configuracao esta ativa. A variavel
    ``HYDRAIA_NATIVE_CA=0`` permite desabilitar explicitamente o comportamento.
    """
    global _CONFIGURADO

    if _CONFIGURADO:
        return True
    if os.getenv("HYDRAIA_NATIVE_CA", "1").strip().lower() in {"0", "false", "no"}:
        return False

    try:
        import truststore
    except ImportError:
        return False

    truststore.inject_into_ssl()
    _CONFIGURADO = True
    return True
