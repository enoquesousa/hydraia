"""FileCache — cache de arquivos JSON com TTL e escrita atomica.

Por que atomico? Em uma escrita normal (`open + write`) um crash do processo
no meio da operacao deixa o arquivo corrompido — e a proxima leitura recebe
um JSON invalido. Usando rename, garantimos que ou o arquivo antigo permanece,
ou o novo esta completo. Em sistemas POSIX e NTFS, `Path.replace` e atomico.

Suporta validacao opcional via callback `validator(data) -> bool`. Dados que
nao passam na validacao sao tratados como cache miss (e descartados se ja
estiverem no disco), evitando que respostas truncadas/corrompidas sejam
servidas como validas.

Principios aplicados:
  - Single Responsibility: a classe sabe SO persistir/expirar pares chave-valor.
  - Resiliencia: cache corrompido ou invalido e descartado em vez de propagar erro.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

Validator = Callable[[Any], bool]


class FileCache:
    """Cache local em arquivos JSON com TTL configuravel e escrita atomica."""

    def __init__(
        self,
        dir: Path,
        ttl_s: int = 3600,
        *,
        validator: Optional[Validator] = None,
    ) -> None:
        self.dir = Path(dir)
        self.ttl_s = ttl_s
        self.validator = validator
        self._lock = threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def get(self, name: str) -> Optional[Any]:
        """Le do cache. Retorna None se ausente, expirado, corrompido ou invalido."""
        path = self._path(name)
        if not path.exists():
            return None

        idade = time.time() - path.stat().st_mtime
        if idade > self.ttl_s:
            logger.info("[cache] %s expirado (%.0fs > %ds) — descartando.",
                        name, idade, self.ttl_s)
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("[cache] %s corrompido (%s) — invalidando.", name, e)
            path.unlink(missing_ok=True)
            return None

        if self.validator is not None and not self.validator(data):
            logger.warning("[cache] %s invalido (validator rejeitou) — invalidando.", name)
            path.unlink(missing_ok=True)
            return None

        logger.info("[cache] %s valido — usando dados em cache.", name)
        return data

    def set(self, name: str, data: Any) -> bool:
        """Persiste no cache via write-then-rename (atomico).

        Retorna True se gravou, False se foi rejeitado pelo validator.
        """
        if self.validator is not None and not self.validator(data):
            logger.warning("[cache] %s rejeitado pelo validator — nao persistindo.", name)
            return False
        target = self._path(name)
        tmp = target.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(target)
        return True

    def invalidate(self, name: str) -> None:
        """Remove uma entrada especifica do cache."""
        self._path(name).unlink(missing_ok=True)


# ─── Validators reusaveis ────────────────────────────────────────────────────

def validador_sabesp(data: Any) -> bool:
    """Aceita dado SABESP so se tiver `sistemas` nao-vazio com volume_percentual."""
    if not isinstance(data, dict):
        return False
    sistemas = data.get("sistemas")
    if not isinstance(sistemas, dict) or not sistemas:
        return False
    # pelo menos um sistema com volume_percentual numerico
    for s in sistemas.values():
        if isinstance(s, dict) and isinstance(s.get("volume_percentual"), (int, float)):
            return True
    return False


def validador_openweather(data: Any) -> bool:
    """Aceita dado OpenWeather so se tiver `previsoes` nao-vazio."""
    if not isinstance(data, dict):
        return False
    previsoes = data.get("previsoes")
    return isinstance(previsoes, list) and len(previsoes) > 0
