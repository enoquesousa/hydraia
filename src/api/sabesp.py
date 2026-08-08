"""
Coletor de dados da API SABESP
==============================
Fonte oficial: https://mananciais.sabesp.com.br/api/v4/

Endpoints reais (descobertos via bundle JS do site):
  GET /api/v4/dados/ultima-data          → ultima data com dados publicados
  GET /api/v4/sistemas/dados/resumo-diario/{data}  → resumo de todos os sistemas

Resiliencia:
  * Circuit breaker — bloqueia chamadas apos N falhas consecutivas
  * Validacao de schema — rejeita resposta com campos faltando / tipos errados
  * Snapshot persistente — `coletar_macrometropole_completa(allow_snapshot=True)`
    cai para o ultimo snapshot valido se a API estiver fora do ar
  * Modo simulado — `coletar_simulado()` para demos offline (TCC)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.utils.paths import HydraPaths

logger = logging.getLogger(__name__)

SISTEMAS_IDS = {
    "cantareira": 64,
    "alto_tiete": 65,
    "guarapiranga": 66,
    "cotia": 67,
    "rio_grande": 68,
    "rio_claro": 69,
    "sao_lourenco": 72,
}

ID_TO_SISTEMA = {v: k for k, v in SISTEMAS_IDS.items()}

SISTEMAS_SABESP = {
    "cantareira": "Sistema Cantareira",
    "alto_tiete": "Sistema Alto Tiete",
    "rio_claro": "Sistema Rio Claro",
    "rio_grande": "Sistema Rio Grande",
    "guarapiranga": "Sistema Guarapiranga",
    "cotia": "Sistema Cotia",
    "sao_lourenco": "Sistema Sao Lourenco",
}

TODOS_SISTEMAS = list(SISTEMAS_SABESP.keys())

SNAPSHOT_NOME = "ultimo_snapshot_valido.json"


# ─── Helpers de validacao de schema ──────────────────────────────────────────

class SchemaError(ValueError):
    """Resposta da API SABESP nao atende ao schema esperado."""


_CAMPOS_OBRIGATORIOS_ITEM = (
    "idSistema",
    "volumeUtilArmazenadoPorcentagem",
)

_CAMPOS_NUMERICOS_OPCIONAIS = (
    "volumeUtilArmazenadoHm3",
    "volumeTotalArmazenado",
    "chuva",
    "vazaoCaptada",
    "vazaoProduzida",
    "vazaoJusante",
    "vazaoNatural",
    "chuvaAcumuladaNoMes",
)


def _validar_resposta_ultima_data(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise SchemaError("ultima-data: payload nao e dict")
    if not payload.get("succeeded"):
        raise SchemaError("ultima-data: succeeded != true")
    data = payload.get("data")
    if not isinstance(data, str) or len(data) < 8:
        raise SchemaError(f"ultima-data: campo data invalido ({data!r})")
    # tenta parse YYYY-MM-DD
    try:
        datetime.strptime(data[:10], "%Y-%m-%d")
    except ValueError as e:
        raise SchemaError(f"ultima-data: data nao parseavel ({data!r})") from e
    return data


def _validar_resposta_resumo(payload: Any) -> List[Dict]:
    if not isinstance(payload, dict):
        raise SchemaError("resumo: payload nao e dict")
    items = payload.get("data")
    if not isinstance(items, list) or not items:
        raise SchemaError("resumo: campo data ausente ou vazio")

    validos: List[Dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("[SABESP] item %d nao e dict — ignorando", i)
            continue
        falt = [c for c in _CAMPOS_OBRIGATORIOS_ITEM if item.get(c) is None]
        if falt:
            logger.warning("[SABESP] item %d sem campos obrigatorios %s — ignorando", i, falt)
            continue
        # normaliza numericos opcionais (string -> float)
        for campo in _CAMPOS_NUMERICOS_OPCIONAIS:
            v = item.get(campo)
            if isinstance(v, str):
                try:
                    item[campo] = float(v.replace(",", "."))
                except ValueError:
                    item[campo] = None
        validos.append(item)

    if not validos:
        raise SchemaError("resumo: nenhum item valido apos validacao")
    return validos


# ─── Coletor ─────────────────────────────────────────────────────────────────

class SabespCollector:
    """Coleta dados REAIS em tempo real da API oficial SABESP, com fallback
    para snapshot local e modo simulado.
    """

    BASE_URL = "https://mananciais.sabesp.com.br/api/v4"

    def __init__(
        self,
        timeout: int = 20,
        retry_attempts: int = 3,
        data_dir: Optional[Path] = None,
        auto_save: bool = True,
        breaker_threshold: int = 4,
        breaker_recovery_s: float = 120.0,
    ):
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.auto_save = auto_save
        self.data_dir = data_dir or HydraPaths.SABESP
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.data_dir / SNAPSHOT_NOME
        self._breaker = CircuitBreaker(
            nome="SABESP",
            failure_threshold=breaker_threshold,
            recovery_s=breaker_recovery_s,
        )

    # ── HTTP base ──

    def _fazer_requisicao(self, endpoint: str) -> Optional[Dict]:
        url = f"{self.BASE_URL}{endpoint}"
        ultimo_erro: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                def _get():
                    return requests.get(url, timeout=self.timeout)
                response = self._breaker.call(_get)
                response.raise_for_status()
                return response.json()
            except CircuitOpenError as e:
                logger.error("[SABESP] %s — abortando.", e)
                return None
            except Exception as e:
                ultimo_erro = e
                logger.warning("[SABESP] Falha na requisicao %s (tentativa %d/%d): %s",
                               url, attempt + 1, self.retry_attempts, e)
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
        if ultimo_erro:
            logger.error("[SABESP] Esgotadas %d tentativas: %s", self.retry_attempts, ultimo_erro)
        return None

    def _obter_ultima_data(self) -> Optional[str]:
        resp = self._fazer_requisicao("/dados/ultima-data")
        if resp is None:
            return None
        try:
            return _validar_resposta_ultima_data(resp)
        except SchemaError as e:
            logger.error("[SABESP] %s", e)
            return None

    # ── Normalizacao ──

    def _normalizar_item(self, item: Dict, data: str) -> Dict:
        return {
            "fonte": "SABESP (Oficial)",
            "sistema": SISTEMAS_SABESP.get(ID_TO_SISTEMA.get(item.get("idSistema"), ""), ""),
            "data": data,
            "volume_percentual": item.get("volumeUtilArmazenadoPorcentagem"),
            "volume_hm3": item.get("volumeUtilArmazenadoHm3"),
            "volume_total_hm3": item.get("volumeTotalArmazenado"),
            "chuva_mm": item.get("chuva"),
            "vazao_captada_m3s": item.get("vazaoCaptada"),
            "vazao_produzida_m3s": item.get("vazaoProduzida"),
            "vazao_jusante_m3s": item.get("vazaoJusante"),
            "vazao_natural_m3s": item.get("vazaoNatural"),
            "chuva_acumulada_mes_mm": item.get("chuvaAcumuladaNoMes"),
            "timestamp_coleta": datetime.now().isoformat(),
        }

    # ── Pipeline principal ──

    def coletar_macrometropole_completa(
        self,
        *,
        allow_snapshot: bool = False,
    ) -> Dict:
        """Coleta todos os sistemas para a ultima data publicada.

        Args:
            allow_snapshot: se True, em caso de falha tenta carregar o ultimo
                snapshot persistido. Default False mantem o comportamento
                "estritamente real" original — escolha explicita do chamador.
        """
        logger.info("[SABESP] Consultando a ultima data disponivel...")
        ultima_data = self._obter_ultima_data()
        if not ultima_data:
            return self._fallback_ou_falha(allow_snapshot, "API SABESP indisponivel para ultima-data")

        logger.info("[SABESP] Ultima data publicada: %s. Coletando dados...", ultima_data)
        resp = self._fazer_requisicao(f"/sistemas/dados/resumo-diario/{ultima_data}")
        if resp is None:
            return self._fallback_ou_falha(
                allow_snapshot, f"API SABESP indisponivel para resumo-diario/{ultima_data}"
            )

        try:
            items = _validar_resposta_resumo(resp)
        except SchemaError as e:
            logger.error("[SABESP] Schema invalido: %s", e)
            return self._fallback_ou_falha(allow_snapshot, str(e))

        resultado = self._montar_resultado(items, ultima_data)
        if self.auto_save:
            self._salvar_coleta(resultado, ultima_data)
            self._atualizar_snapshot(resultado)
        return resultado

    def _montar_resultado(self, items: List[Dict], ultima_data: str) -> Dict:
        resultado: Dict = {
            "fonte": "SABESP (Oficial)",
            "timestamp_coleta": datetime.now().isoformat(),
            "data_referencia": ultima_data,
            "sistemas": {},
            "resumo": {},
            "modo": "real",
        }
        volumes: List[float] = []
        for item in items:
            chave = ID_TO_SISTEMA.get(item.get("idSistema"))
            if not chave:
                continue
            norm = self._normalizar_item(item, ultima_data)
            resultado["sistemas"][chave] = norm
            vol = norm.get("volume_percentual")
            if isinstance(vol, (int, float)):
                volumes.append(float(vol))

        if volumes:
            resultado["resumo"] = {
                "total_sistemas": len(volumes),
                "volume_medio_percentual": round(sum(volumes) / len(volumes), 2),
                "sistemas_abaixo_50pct": sum(1 for v in volumes if v < 50),
                "sistemas_criticos": sum(1 for v in volumes if v < 30),
            }

        logger.info("[SABESP] OK: %d sistemas para %s.", len(resultado["sistemas"]), ultima_data)
        return resultado

    # ── Snapshot e fallback ──

    def _fallback_ou_falha(self, allow_snapshot: bool, motivo: str) -> Dict:
        if allow_snapshot:
            snap = self._carregar_snapshot()
            if snap is not None:
                snap["modo"] = "snapshot"
                snap["fallback_motivo"] = motivo
                logger.warning("[SABESP] Usando ultimo snapshot persistido (motivo: %s)", motivo)
                return snap
        raise RuntimeError(f"❌ Coleta SABESP falhou: {motivo}")

    def _carregar_snapshot(self) -> Optional[Dict]:
        if not self.snapshot_path.exists():
            return None
        try:
            return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[SABESP] Snapshot corrompido (%s) — descartando", e)
            return None

    def _atualizar_snapshot(self, dados: Dict) -> None:
        tmp = self.snapshot_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.snapshot_path)
        except OSError as e:
            logger.warning("[SABESP] Falha ao atualizar snapshot: %s", e)

    # ── Persistencia bruta ──

    def _salvar_coleta(self, dados: Dict, data_ref: str):
        pasta = self.data_dir / data_ref
        pasta.mkdir(parents=True, exist_ok=True)
        path = pasta / f"coleta_real_{datetime.now().strftime('%H%M%S')}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)
            logger.info("[SABESP] Salvo em %s", path)
        except OSError as e:
            logger.warning("[SABESP] Falha ao salvar coleta bruta: %s", e)

    # ── Modo simulado (offline / TCC demo) ──

    def coletar_simulado(self, data_ref: Optional[str] = None) -> Dict:
        """Gera dados sinteticos plausiveis para demos offline / dry-run.

        Util quando a API SABESP esta fora do ar e nao ha snapshot, ou para
        testes de pipeline sem trafego real.
        """
        if data_ref is None:
            data_ref = (date.today() - timedelta(days=1)).isoformat()

        # Volumes plausiveis (faixa de atencao a normal) baseados em historico
        valores = {
            "cantareira":   {"vol": 52.4, "chuva": 1.2, "v_natural": 18.5, "v_jusante": 5.0},
            "alto_tiete":   {"vol": 60.8, "chuva": 0.8, "v_natural": 9.4,  "v_jusante": 3.1},
            "guarapiranga": {"vol": 78.6, "chuva": 1.0, "v_natural": 14.2, "v_jusante": 6.2},
            "cotia":        {"vol": 41.2, "chuva": 0.6, "v_natural": 1.8,  "v_jusante": 0.9},
            "rio_grande":   {"vol": 71.5, "chuva": 0.9, "v_natural": 8.6,  "v_jusante": 4.0},
            "rio_claro":    {"vol": 65.3, "chuva": 1.1, "v_natural": 4.7,  "v_jusante": 2.4},
            "sao_lourenco": {"vol": 80.1, "chuva": 1.4, "v_natural": 5.9,  "v_jusante": 2.8},
        }

        sistemas: Dict[str, Dict] = {}
        volumes: List[float] = []
        for chave, v in valores.items():
            sistemas[chave] = {
                "fonte": "SABESP (Simulado)",
                "sistema": SISTEMAS_SABESP[chave],
                "data": data_ref,
                "volume_percentual": v["vol"],
                "volume_hm3": round(v["vol"] * 12.0, 1),
                "volume_total_hm3": round(v["vol"] * 14.0, 1),
                "chuva_mm": v["chuva"],
                "vazao_captada_m3s": v["v_jusante"] * 1.5,
                "vazao_produzida_m3s": v["v_jusante"] * 1.4,
                "vazao_jusante_m3s": v["v_jusante"],
                "vazao_natural_m3s": v["v_natural"],
                "chuva_acumulada_mes_mm": v["chuva"] * 30,
                "timestamp_coleta": datetime.now().isoformat(),
            }
            volumes.append(v["vol"])

        resultado = {
            "fonte": "SABESP (Simulado)",
            "timestamp_coleta": datetime.now().isoformat(),
            "data_referencia": data_ref,
            "sistemas": sistemas,
            "resumo": {
                "total_sistemas": len(volumes),
                "volume_medio_percentual": round(sum(volumes) / len(volumes), 2),
                "sistemas_abaixo_50pct": sum(1 for v in volumes if v < 50),
                "sistemas_criticos": sum(1 for v in volumes if v < 30),
            },
            "modo": "simulado",
        }
        logger.warning("[SABESP] Coleta SIMULADA (offline) para %s — %d sistemas.",
                       data_ref, len(sistemas))
        return resultado


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    col = SabespCollector()
    print(col.coletar_macrometropole_completa(allow_snapshot=True))
