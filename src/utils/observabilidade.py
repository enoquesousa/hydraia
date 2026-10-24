"""Observabilidade — logging estruturado, correlation IDs, metricas e versionamento.

Componentes:

  * `JsonFormatter` — formata logs como JSON line (uma linha por registro).
  * `CorrelationFilter` — injeta `correlation_id` (contextvar) em todo log da
    mesma requisicao/pipeline run.
  * `MetricsRegistry` — coleta latencias, contadores e estados (in-process).
  * `versao_codigo()` — captura hash do diretorio src/ para versionamento de
    relatorios (sem precisar de git).
  * `mascarar_token` — utilitario reusavel para apagar chaves em logs.

Uso minimo:

    from src.utils.observabilidade import setup_logging, novo_correlation_id, METRICS

    setup_logging()              # uma vez no boot
    novo_correlation_id()        # inicio de cada execucao
    with METRICS.timer("etapa.x"): ...
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, Optional
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit


# ─── Correlation ID ──────────────────────────────────────────────────────────

_CORRELATION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hydraia_correlation_id", default="-"
)


def novo_correlation_id(prefix: str = "run") -> str:
    cid = f"{prefix}-{uuid.uuid4().hex[:12]}"
    _CORRELATION_ID.set(cid)
    return cid


def correlation_id() -> str:
    return _CORRELATION_ID.get()


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id()
        return True


# ─── Mascaramento de tokens ──────────────────────────────────────────────────

_RE_TOKEN = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{10,}|AIza[A-Za-z0-9_\-]{20,}|"
    r"(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=\-]+)", re.IGNORECASE,
)
_RE_TOKEN_KV = re.compile(
    r'''(?<![\w%\-])(?P<key>[\w%\-]+)(?P<separator>["']?\s*[:=]\s*)'''
    r'''(?P<value>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s"',;&}\]]+)''',
    re.IGNORECASE,
)
_RE_URL = re.compile(r'''https?://[^\s<>"']+''', re.IGNORECASE)
_SENSITIVE_FIELDS = {
    "appid", "key", "api_key", "apikey", "token", "authorization",
    "proxy_authorization", "bearer", "secret", "password", "accesstoken",
    "refreshtoken", "idtoken",
}


def _mascarar_campo(match: re.Match) -> str:
    key = unquote(match.group("key")).lower().replace("-", "_")
    sensitive = key in _SENSITIVE_FIELDS or key.endswith(
        ("_api_key", "_token", "_secret", "_password")
    )
    if not sensitive:
        return match.group(0)
    value = match.group("value")
    quote = value[0] if value[0] in "\"'" else ""
    return f"{match.group('key')}{match.group('separator')}{quote}***{quote}"


def _mascarar_url(match: re.Match) -> str:
    """Mantem endpoint e nomes dos parametros, nunca seus valores."""
    try:
        parts = urlsplit(match.group(0))
        query = urlencode(
            [(name, "***") for name, _ in parse_qsl(
                parts.query, keep_blank_values=True, max_num_fields=256,
            )],
            safe="*",
        )
        netloc = parts.netloc
        if "@" in netloc:
            netloc = "***@" + netloc.rsplit("@", 1)[1]
        return urlunsplit((
            parts.scheme, netloc, parts.path, query, "***" if parts.fragment else "",
        ))
    except ValueError:
        return "[URL omitida]"


def mascarar_token(texto: str) -> str:
    """Oculta queries e padroes conhecidos; nao reconhece todo segredo arbitrario."""
    if not texto:
        return texto
    out = _RE_TOKEN.sub("***", texto)
    out = _RE_TOKEN_KV.sub(_mascarar_campo, out)
    return _RE_URL.sub(_mascarar_url, out)


class MaskingFilter(logging.Filter):
    """Mascara tokens em mensagens de log antes de elas serem formatadas."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            record.msg = mascarar_token(msg)
            record.args = ()
        except Exception:
            # Nao deixe o erro de interpolacao do logging imprimir args brutos.
            record.msg = "Mensagem de log indisponivel."
            record.args = ()
        return True


# ─── Formatadores ────────────────────────────────────────────────────────────

def _exception_lines(error: BaseException, tb, seen: set[int]) -> list[str]:
    if id(error) in seen:
        return ["[referencia de excecao repetida]"]
    seen.add(id(error))
    lines = []
    previous = error.__cause__
    if previous is None and not error.__suppress_context__:
        previous = error.__context__
    if previous is not None:
        lines.extend(_exception_lines(previous, previous.__traceback__, seen))
        lines.append("A excecao acima antecedeu a seguinte:")
    if tb is not None:
        lines.append("Traceback (most recent call last):")
        for frame, lineno in traceback.walk_tb(tb):
            # Nao consulte linhas de fonte, variaveis locais ou str(error).
            filename = json.dumps(frame.f_code.co_filename, ensure_ascii=False)
            lines.append(f"  File {filename}, line {lineno}, in {frame.f_code.co_name}")
    lines.append(f"{type(error).__name__}: [mensagem omitida]")
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            lines.extend(_exception_lines(child, child.__traceback__, seen))
    return lines


class _SafeFormatter(logging.Formatter):
    """Preserva classe/frames sem serializar valores potencialmente confidenciais."""

    def formatException(self, exc_info) -> str:
        try:
            return mascarar_token("\n".join(_exception_lines(exc_info[1], exc_info[2], set())))
        except Exception:
            return "Detalhes da excecao indisponiveis."

    def formatStack(self, stack_info: str) -> str:
        frames = [line for line in stack_info.splitlines() if line.startswith('  File "')]
        return mascarar_token("\n".join(["Stack (most recent call last):", *frames]))

    def format(self, record: logging.LogRecord) -> str:
        # Outro handler pode ter armazenado uma representacao nao sanitizada.
        record.exc_text = None
        return mascarar_token(super().format(record))


class JsonFormatter(_SafeFormatter):
    """Formata logs como JSON line — facil de parsear/grep em pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": mascarar_token(record.name),
            "correlation_id": mascarar_token(getattr(record, "correlation_id", "-")),
            "message": mascarar_token(record.getMessage()),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


# ─── Setup ───────────────────────────────────────────────────────────────────

class _DynamicStderr:
    """Encaminha cada escrita ao stderr corrente sem reter capturas temporarias."""

    def write(self, texto: str) -> int:
        return sys.stderr.write(texto)

    def flush(self) -> None:
        sys.stderr.flush()


def setup_logging(
    *,
    nivel: int = logging.INFO,
    json_logs: Optional[bool] = None,
    arquivo: Optional[Path] = None,
) -> None:
    """Configura logging. Se `json_logs` for None, le do env `HYDRAIA_JSON_LOGS`."""
    if json_logs is None:
        json_logs = os.environ.get("HYDRAIA_JSON_LOGS", "").lower() in {"1", "true", "yes"}

    root = logging.getLogger()
    root.setLevel(nivel)
    # Limpa handlers antigos para nao duplicar
    for h in list(root.handlers):
        root.removeHandler(h)

    # O proxy preserva captura de logs em testes/cadernos sem reter um stream
    # temporario que pode ser fechado antes do proximo ciclo de logging.
    handler: logging.Handler = logging.StreamHandler(_DynamicStderr())
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(_SafeFormatter(
            "%(asctime)s [%(levelname)s] [cid=%(correlation_id)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
    handler.addFilter(MaskingFilter())
    handler.addFilter(CorrelationFilter())
    root.addHandler(handler)

    if arquivo is not None:
        try:
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(arquivo, encoding="utf-8")
            fh.setFormatter(JsonFormatter())
            fh.addFilter(MaskingFilter())
            fh.addFilter(CorrelationFilter())
            root.addHandler(fh)
        except OSError:
            pass


# ─── Metricas in-process ─────────────────────────────────────────────────────

@dataclass
class _Counter:
    valor: int = 0


@dataclass
class _Histograma:
    valores: list = field(default_factory=list)

    def add(self, v: float) -> None:
        self.valores.append(float(v))

    def stats(self) -> Dict[str, float]:
        if not self.valores:
            return {"n": 0}
        vs = sorted(self.valores)
        n = len(vs)
        return {
            "n": n,
            "min": vs[0],
            "max": vs[-1],
            "mean": sum(vs) / n,
            "p50": vs[n // 2],
            "p95": vs[int(n * 0.95)] if n >= 20 else vs[-1],
        }


class MetricsRegistry:
    """Registro thread-safe de contadores e histogramas (latencias)."""

    def __init__(self) -> None:
        self._counters: Dict[str, _Counter] = {}
        self._hist: Dict[str, _Histograma] = {}
        self._lock = RLock()

    def inc(self, nome: str, n: int = 1) -> None:
        with self._lock:
            self._counters.setdefault(nome, _Counter()).valor += n

    def observe(self, nome: str, valor: float) -> None:
        with self._lock:
            self._hist.setdefault(nome, _Histograma()).add(valor)

    @contextmanager
    def timer(self, nome: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(f"{nome}.latency_s", time.perf_counter() - t0)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": {k: c.valor for k, c in self._counters.items()},
                "histograms": {k: h.stats() for k, h in self._hist.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._hist.clear()


METRICS = MetricsRegistry()


# ─── Versionamento ───────────────────────────────────────────────────────────

_VERSAO_CACHE: Dict[str, tuple[float, str]] = {}


def versao_codigo(src_dir: Optional[Path] = None, *, fast: bool = False) -> str:
    """Hash sha256 truncado (12 chars) das fontes em `src/`.

    Por default le o conteudo dos arquivos (`fast=False`) para nao colidir em
    edits rapidos que preservem mtime/size. Em runs muito grandes, o caller
    pode optar por `fast=True` (mtime+size) para evitar I/O pesado.

    Cache em memoria por `src_dir` + 5s — segunda chamada na mesma execucao
    nao re-le todos os arquivos.
    """
    if src_dir is None:
        src_dir = Path(__file__).resolve().parent.parent
    chave = f"{src_dir}:{fast}"
    agora = time.time()
    cached = _VERSAO_CACHE.get(chave)
    if cached and (agora - cached[0]) < 5.0:
        return cached[1]

    h = hashlib.sha256()
    arquivos = sorted(src_dir.rglob("*.py"))
    for p in arquivos:
        try:
            h.update(str(p.relative_to(src_dir)).encode("utf-8"))
            if fast:
                st = p.stat()
                h.update(str(int(st.st_mtime)).encode("utf-8"))
                h.update(str(st.st_size).encode("utf-8"))
            else:
                with open(p, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
        except OSError:
            continue
    out = h.hexdigest()[:12]
    _VERSAO_CACHE[chave] = (agora, out)
    return out


def hash_prompt(prompt: str) -> str:
    """Hash sha256 truncado de um prompt — versionamento de relatorios."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
