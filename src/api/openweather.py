"""
Coletor de dados da API OpenWeatherMap
Documentacao: https://openweathermap.org/api

Resiliencia:
  * Circuit breaker — bloqueia chamadas apos N falhas consecutivas
  * Validacao de schema — rejeita resposta com campos faltando/tipos errados
  * Modo simulado — `coletar_simulado()` para demos offline (TCC)
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import requests

from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.utils.env_loader import EnvLoader
from src.utils.http_security import exigir_resposta_sem_redirecionamento, validar_url_servico

logger = logging.getLogger(__name__)


class WeatherSchemaError(ValueError):
    """Resposta da OpenWeather nao atende ao schema esperado."""


def _validar_schema_one_call(payload: Any) -> Dict:
    if not isinstance(payload, dict):
        raise WeatherSchemaError("payload nao e dict")
    if "daily" not in payload or not isinstance(payload["daily"], list):
        raise WeatherSchemaError("campo 'daily' ausente ou nao-lista")
    if not payload["daily"]:
        raise WeatherSchemaError("'daily' vazio — sem previsoes")
    return payload


def _diagnostico_http(error: requests.RequestException) -> tuple[str, int | str]:
    """Use apenas categorias conhecidas e status; nunca str(error), URL ou body."""
    classes = (
        requests.exceptions.SSLError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
        requests.exceptions.TooManyRedirects,
        requests.exceptions.InvalidURL,
    )
    category = next((kind.__name__ for kind in classes if isinstance(error, kind)),
                    "RequestException")
    status = getattr(error.response, "status_code", None)
    if type(status) is not int or not 100 <= status <= 599:
        status = "indisponivel"
    return category, status


class OpenWeatherCollector:
    """Coleta previsoes meteorologicas da OpenWeatherMap API"""

    def __init__(self, config: Dict):
        """
        Args:
            config: Dicionario com 'base_url', 'timeout', 'retry_attempts'
        """
        self.api_key = EnvLoader.get('OPENWEATHER_API_KEY')

        if not self.api_key:
            raise ValueError(
                "OpenWeather API key nao encontrada no .env\n"
                "Obtenha uma chave gratuita em: https://openweathermap.org/api\n"
                "Configure OPENWEATHER_API_KEY no arquivo .env"
            )

        self.base_url = validar_url_servico(
            config.get('base_url', 'https://api.openweathermap.org/data/3.0')
        )
        self.timeout = config.get('timeout', 10)
        self.retry_attempts = config.get('retry_attempts', 3)
        self._breaker = CircuitBreaker(
            nome="OpenWeather",
            failure_threshold=int(config.get("breaker_threshold", 4)),
            recovery_s=float(config.get("breaker_recovery_s", 120.0)),
        )

    def coletar_previsao(self, lat: float, lon: float, dias: int = 7) -> Optional[Dict]:
        """Coleta previsao para uma localizacao geografica.

        Args:
            lat: Latitude
            lon: Longitude
            dias: Numero de dias (max 8 no plano free)

        Returns:
            Dicionario normalizado ou None em caso de erro / circuit aberto.
        """
        endpoint = f"{self.base_url}/onecall"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'pt_br',
            'exclude': 'minutely,alerts',
        }

        for attempt in range(self.retry_attempts):
            try:
                def _get():
                    return requests.get(
                        endpoint, params=params, timeout=self.timeout, allow_redirects=False,
                    )
                response = self._breaker.call(_get)
                try:
                    exigir_resposta_sem_redirecionamento(response)
                except RuntimeError:
                    logger.error("[OpenWeather] Redirecionamento HTTP recusado.")
                    return None
                response.raise_for_status()
                data = _validar_schema_one_call(response.json())
                return self._processar_resposta(data)

            except CircuitOpenError as e:
                logger.error("[OpenWeather] %s — abortando.", e)
                return None
            except WeatherSchemaError as e:
                logger.error("[OpenWeather] Schema invalido: %s", e)
                return None
            except requests.exceptions.RequestException as e:
                category, status = _diagnostico_http(e)
                logger.warning("[OpenWeather] Tentativa %d/%d falhou (%s; HTTP %s).",
                               attempt + 1, self.retry_attempts, category, status)
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("[OpenWeather] Erro apos %d tentativas", self.retry_attempts)
                    return None
        return None

    def _processar_resposta(self, data: Dict) -> Dict:
        previsoes: List[Dict] = []

        for item in data.get('daily', []):
            timestamp = item.get('dt')
            try:
                dt = datetime.fromtimestamp(timestamp)
            except (TypeError, OSError, ValueError):
                continue

            temp_info = item.get('temp', {}) or {}

            previsao = {
                'fonte': 'OpenWeather',
                'timestamp': timestamp,
                'data_hora': dt.isoformat(),
                'temperatura_c': temp_info.get('day'),
                'sensacao_termica_c': (item.get('feels_like') or {}).get('day'),
                'umidade_pct': item.get('humidity'),
                'pressao_hpa': item.get('pressure'),
                'precipitacao_mm': item.get('rain', 0.0) or 0.0,
                'vento_vel_ms': item.get('wind_speed'),
                'vento_direcao_graus': item.get('wind_deg', 0),
                'nuvens_pct': item.get('clouds'),
                'descricao': (item.get('weather') or [{}])[0].get('description'),
                'condicao_codigo': (item.get('weather') or [{}])[0].get('id'),
            }
            previsoes.append(previsao)

        if not previsoes:
            raise WeatherSchemaError("nenhuma previsao diaria parseavel")

        current_raw = data.get('current') or {}
        current: Optional[Dict] = None
        if current_raw:
            current = {
                'dt': current_raw.get('dt'),
                'temp': current_raw.get('temp'),
                'feels_like': current_raw.get('feels_like'),
                'humidity': current_raw.get('humidity'),
                'pressure': current_raw.get('pressure'),
                'uvi': current_raw.get('uvi'),
                'wind_speed': current_raw.get('wind_speed'),
                'wind_deg': current_raw.get('wind_deg'),
                'clouds': current_raw.get('clouds'),
                'rain_1h': (current_raw.get('rain') or {}).get('1h', 0.0),
                'snow_1h': (current_raw.get('snow') or {}).get('1h', 0.0),
                'description': (current_raw.get('weather') or [{}])[0].get('description'),
                'condition_id': (current_raw.get('weather') or [{}])[0].get('id'),
            }

        return {
            'fonte': 'OpenWeather',
            'coordenadas': {
                'lat': data.get('lat'),
                'lon': data.get('lon'),
            },
            'current': current,
            'previsoes': previsoes,
            'total_previsoes': len(previsoes),
            'coleta_timestamp': datetime.now().isoformat(),
            'modo': 'real',
        }

    def calcular_acumulado_diario(self, previsoes: List[Dict]) -> List[Dict]:
        """Agrupa previsoes em acumulados diarios"""
        dias: Dict[str, Dict] = {}

        for prev in previsoes:
            data = prev['data_hora'].split('T')[0]
            if data not in dias:
                dias[data] = {
                    'data': data,
                    'precipitacao_total_mm': 0.0,
                    'temp_media_c': [],
                    'temp_max_c': -999,
                    'temp_min_c': 999,
                    'umidade_media_pct': [],
                }

            dias[data]['precipitacao_total_mm'] += prev['precipitacao_mm']
            dias[data]['temp_media_c'].append(prev['temperatura_c'])
            dias[data]['temp_max_c'] = max(dias[data]['temp_max_c'], prev['temperatura_c'])
            dias[data]['temp_min_c'] = min(dias[data]['temp_min_c'], prev['temperatura_c'])
            dias[data]['umidade_media_pct'].append(prev['umidade_pct'])

        resultado = []
        for data, info in sorted(dias.items()):
            info['temp_media_c'] = sum(info['temp_media_c']) / len(info['temp_media_c'])
            info['umidade_media_pct'] = sum(info['umidade_media_pct']) / len(info['umidade_media_pct'])
            resultado.append(info)

        return resultado


def coletar_simulado(dias: int = 7) -> Dict:
    """Previsoes meteorologicas sinteticas para demos offline."""
    base = datetime.now()
    previsoes = []
    for i in range(dias):
        d = base + timedelta(days=i)
        previsoes.append({
            'fonte': 'OpenWeather (Simulado)',
            'timestamp': int(d.timestamp()),
            'data_hora': d.isoformat(),
            'temperatura_c': 22.0 + (i % 3),
            'sensacao_termica_c': 22.5 + (i % 3),
            'umidade_pct': 70 + (i % 5),
            'pressao_hpa': 1015,
            'precipitacao_mm': 5.0 if i % 2 else 0.0,
            'vento_vel_ms': 3.2,
            'vento_direcao_graus': 180,
            'nuvens_pct': 60,
            'descricao': 'parcialmente nublado',
            'condicao_codigo': 802,
        })
    return {
        'fonte': 'OpenWeather (Simulado)',
        'coordenadas': {'lat': -23.5505, 'lon': -46.6333},
        'current': {
            'temp': 22.0, 'humidity': 72, 'pressure': 1015,
            'rain_1h': 0.0, 'description': 'parcialmente nublado',
        },
        'previsoes': previsoes,
        'total_previsoes': len(previsoes),
        'coleta_timestamp': datetime.now().isoformat(),
        'modo': 'simulado',
    }


if __name__ == "__main__":
    import json
    from src.utils.paths import HydraPaths

    with open(HydraPaths.BASE / "config_apis.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    coletor = OpenWeatherCollector(config["openweather"])
    dados = coletor.coletar_previsao(-23.5505, -46.6333, dias=5)

    if dados:
        coords = dados.get("coordenadas", {})
        print(f"OK Coletadas {dados['total_previsoes']} previsoes "
              f"em lat={coords.get('lat')}, lon={coords.get('lon')}")
    else:
        print("ERRO ao coletar dados")
