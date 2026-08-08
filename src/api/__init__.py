"""
Coletores de APIs ativos no pipeline HydraIA.
=============================================
- SabespCollector  → reservatórios da Macrometrópole (API oficial v4)
- OpenWeatherCollector → previsão meteorológica (One Call 3.0)
"""

from .openweather import OpenWeatherCollector
from .sabesp import SabespCollector, SISTEMAS_SABESP, TODOS_SISTEMAS

__all__ = [
    "OpenWeatherCollector",
    "SabespCollector",
    "SISTEMAS_SABESP",
    "TODOS_SISTEMAS",
]
