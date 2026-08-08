"""Baselines de previsao temporal — referencia para validar o HydroLSTM.

Um modelo so vale o que ganha sobre baselines simples. Aqui implementamos:

  * Persistence — "amanha = hoje" (replicado para todo o horizonte).
  * Sazonal mensal — media historica do mesmo mes (climatologia).
  * Drift linear — extrapola tendencia recente.

Cada baseline expoe `prever(janela: np.ndarray, horizon: int) -> np.ndarray`.

Uso:
    from src.data.baselines import (
        PersistenceBaseline, SazonalMensalBaseline,
        comparar_baselines,
    )
    metrics = comparar_baselines(y_pred_lstm, y_true, X_janela_test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class BaselineMetrics:
    nome: str
    mae: float
    rmse: float

    def __str__(self) -> str:
        return f"{self.nome:<25} MAE={self.mae:.3f}  RMSE={self.rmse:.3f}"


def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ─── Baselines ───────────────────────────────────────────────────────────────

class Baseline:
    """Classe base — uma instancia por dataset/serie."""

    nome: str = "baseline"

    def prever(self, janela: np.ndarray, horizon: int) -> np.ndarray:
        """janela: (lookback, n_features); retorna (horizon, n_targets)."""
        raise NotImplementedError


class PersistenceBaseline(Baseline):
    """Repete o ultimo valor das colunas-alvo para todo o horizonte."""

    nome = "persistence"

    def __init__(self, target_idx_in_features: list[int]) -> None:
        # indices das colunas-alvo dentro do array de features
        self.tgt_idx = list(target_idx_in_features)

    def prever(self, janela: np.ndarray, horizon: int) -> np.ndarray:
        ultimo = janela[-1, self.tgt_idx]   # (n_targets,)
        return np.tile(ultimo, (horizon, 1))


class SazonalMensalBaseline(Baseline):
    """Media historica por mes e dia-do-mes — climatologia simples."""

    nome = "sazonal_mensal"

    def __init__(
        self,
        media_por_mes: np.ndarray,  # (12, n_targets)
        mes_inicial: int,           # 1-12 — mes do primeiro dia previsto
    ) -> None:
        self.media_por_mes = np.asarray(media_por_mes, dtype=np.float32)
        self.mes_inicial = int(mes_inicial)

    def prever(self, janela: np.ndarray, horizon: int) -> np.ndarray:
        out = np.zeros((horizon, self.media_por_mes.shape[1]), dtype=np.float32)
        # aproxima: usa o mes inicial para todos os dias do horizonte
        idx = self.mes_inicial - 1
        out[:] = self.media_por_mes[idx]
        return out


class DriftLinearBaseline(Baseline):
    """Extrapola linearmente a tendencia das ultimas N observacoes."""

    nome = "drift_linear"

    def __init__(self, target_idx_in_features: list[int], n_pontos: int = 14) -> None:
        self.tgt_idx = list(target_idx_in_features)
        self.n_pontos = int(n_pontos)

    def prever(self, janela: np.ndarray, horizon: int) -> np.ndarray:
        n = min(self.n_pontos, janela.shape[0])
        recente = janela[-n:, self.tgt_idx]            # (n, n_targets)
        x = np.arange(n).astype(np.float32)
        # ajusta uma reta por coluna
        slopes = np.array([
            np.polyfit(x, recente[:, i], 1)[0] for i in range(recente.shape[1])
        ], dtype=np.float32)
        intercept = recente[-1]                         # ponto inicial
        out = np.zeros((horizon, recente.shape[1]), dtype=np.float32)
        for h in range(horizon):
            out[h] = intercept + slopes * (h + 1)
        return out


# ─── Comparacao agregada ─────────────────────────────────────────────────────

def comparar_baselines(
    janelas_X: np.ndarray,        # (N, lookback, n_features) (desnormalizado)
    y_true: np.ndarray,           # (N, horizon, n_targets) (desnormalizado)
    target_idx_in_features: list[int],
    *,
    media_por_mes: Optional[np.ndarray] = None,
    mes_inicial_por_amostra: Optional[np.ndarray] = None,
) -> Dict[str, BaselineMetrics]:
    """Avalia cada baseline em todas as janelas e devolve metricas agregadas."""
    horizon = y_true.shape[1]
    baselines: Dict[str, Baseline] = {
        "persistence": PersistenceBaseline(target_idx_in_features),
        "drift_linear": DriftLinearBaseline(target_idx_in_features, n_pontos=14),
    }

    if media_por_mes is not None and mes_inicial_por_amostra is not None:
        # Sazonal exige uma instancia por amostra (mes inicial varia)
        # Para simplificar a interface aqui, calculamos diretamente em loop.
        pass

    resultados: Dict[str, BaselineMetrics] = {}
    for nome, base in baselines.items():
        preds = np.stack(
            [base.prever(janelas_X[i], horizon) for i in range(janelas_X.shape[0])],
            axis=0,
        )
        resultados[nome] = BaselineMetrics(
            nome=nome,
            mae=_mae(preds, y_true),
            rmse=_rmse(preds, y_true),
        )

    if media_por_mes is not None and mes_inicial_por_amostra is not None:
        preds_list = []
        for i in range(janelas_X.shape[0]):
            mes = int(mes_inicial_por_amostra[i])
            base = SazonalMensalBaseline(media_por_mes, mes)
            preds_list.append(base.prever(janelas_X[i], horizon))
        preds = np.stack(preds_list, axis=0)
        resultados["sazonal_mensal"] = BaselineMetrics(
            nome="sazonal_mensal",
            mae=_mae(preds, y_true),
            rmse=_rmse(preds, y_true),
        )

    return resultados


# ─── Backtesting rolling ─────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    janelas_avaliadas: int
    mae_por_horizonte: np.ndarray       # (horizon,)
    rmse_por_horizonte: np.ndarray      # (horizon,)
    mae_global: float
    rmse_global: float

    def resumo(self) -> str:
        lines = [
            f"Backtest rolling — {self.janelas_avaliadas} janelas",
            f"  MAE global  = {self.mae_global:.3f}",
            f"  RMSE global = {self.rmse_global:.3f}",
            "  MAE por horizonte:",
        ]
        for h, m in enumerate(self.mae_por_horizonte, 1):
            lines.append(f"    +{h}d: {m:.3f}")
        return "\n".join(lines)


def backtest_rolling(
    preds: np.ndarray,    # (N, horizon, n_targets)
    y_true: np.ndarray,   # (N, horizon, n_targets)
) -> BacktestResult:
    """Calcula MAE/RMSE por dia do horizonte (deteccao de degradacao temporal)."""
    if preds.shape != y_true.shape:
        raise ValueError(f"shapes diferentes: {preds.shape} vs {y_true.shape}")
    horizon = preds.shape[1]
    mae_h = np.array([
        float(np.mean(np.abs(preds[:, h, :] - y_true[:, h, :])))
        for h in range(horizon)
    ], dtype=np.float32)
    rmse_h = np.array([
        float(np.sqrt(np.mean((preds[:, h, :] - y_true[:, h, :]) ** 2)))
        for h in range(horizon)
    ], dtype=np.float32)
    return BacktestResult(
        janelas_avaliadas=int(preds.shape[0]),
        mae_por_horizonte=mae_h,
        rmse_por_horizonte=rmse_h,
        mae_global=float(np.mean(np.abs(preds - y_true))),
        rmse_global=float(np.sqrt(np.mean((preds - y_true) ** 2))),
    )


# ─── Probabilidade de eventos (apoio a inferencia) ─────────────────────────────

def prob_evento(valor_central: float, residuos: np.ndarray, limite: float, *, direcao: str = "abaixo") -> float:
    """Calcula P(pred + residuo <= limite) ou >= limite.

    Args:
        valor_central: previsão pontual do modelo.
        residuos: array 1D de resíduos (real - previsto) amostrados em validação.
        limite: valor do evento (ex.: 30% de volume).
        direcao: "abaixo" ou "acima".

    Returns:
        Probabilidade em [0,1]. Se não houver resíduos, devolve 0.0.
    """
    if residuos is None or len(residuos) == 0:
        return 0.0
    sims = valor_central + residuos
    if direcao == "acima":
        prob = float(np.mean(sims >= limite))
    else:
        prob = float(np.mean(sims <= limite))
    return max(0.0, min(1.0, prob))


def resumo_probabilidades(valor_central: float, residuos: np.ndarray, limites: list[float]) -> Dict[str, float]:
    """Gera um dicionário com probabilidades abaixo/acima dos limites fornecidos."""
    out: Dict[str, float] = {}
    for lim in limites:
        out[f"prob_abaixo_{int(lim)}"] = prob_evento(valor_central, residuos, lim, direcao="abaixo")
        out[f"prob_acima_{int(lim)}"] = prob_evento(valor_central, residuos, lim, direcao="acima")
    return out
