"""Testes dos baselines temporais e do backtesting rolling."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.baselines import (
    DriftLinearBaseline,
    PersistenceBaseline,
    SazonalMensalBaseline,
    backtest_rolling,
    comparar_baselines,
)


# ─── Persistence ─────────────────────────────────────────────────────────────

class TestPersistence:
    def test_repete_ultimo_valor(self):
        # janela: (5 dias, 4 features). Targets nas posicoes [0, 2]
        janela = np.array([
            [10, 1, 20, 0],
            [11, 1, 21, 0],
            [12, 1, 22, 0],
            [13, 1, 23, 0],
            [14, 1, 24, 0],
        ], dtype=np.float32)
        b = PersistenceBaseline(target_idx_in_features=[0, 2])
        out = b.prever(janela, horizon=3)
        assert out.shape == (3, 2)
        assert np.allclose(out, [[14, 24], [14, 24], [14, 24]])


# ─── Drift ───────────────────────────────────────────────────────────────────

class TestDrift:
    def test_extrapola_tendencia_constante(self):
        # serie crescente +1/dia
        janela = np.array([[i, 0] for i in range(20)], dtype=np.float32)
        b = DriftLinearBaseline(target_idx_in_features=[0], n_pontos=10)
        out = b.prever(janela, horizon=5)
        assert out.shape == (5, 1)
        # ultimo valor 19, slope ~1 -> proximos 5: 20,21,22,23,24
        assert abs(out[0, 0] - 20.0) < 0.1
        assert abs(out[4, 0] - 24.0) < 0.1


# ─── Sazonal ─────────────────────────────────────────────────────────────────

class TestSazonal:
    def test_retorna_media_do_mes(self):
        media = np.zeros((12, 2), dtype=np.float32)
        media[5] = [40.0, 50.0]   # junho
        b = SazonalMensalBaseline(media_por_mes=media, mes_inicial=6)
        out = b.prever(np.zeros((10, 5)), horizon=4)
        assert out.shape == (4, 2)
        assert np.allclose(out, [[40, 50]] * 4)


# ─── comparar_baselines ──────────────────────────────────────────────────────

class TestCompararBaselines:
    def test_retorna_metricas_para_cada_baseline(self):
        N, lb, n_feat = 5, 8, 3
        H, n_tgt = 3, 1
        rng = np.random.default_rng(0)
        X = rng.normal(size=(N, lb, n_feat)).astype(np.float32)
        y = rng.normal(size=(N, H, n_tgt)).astype(np.float32)

        out = comparar_baselines(X, y, target_idx_in_features=[0])
        assert "persistence" in out
        assert "drift_linear" in out
        for m in out.values():
            assert m.mae >= 0
            assert m.rmse >= 0


# ─── Backtest rolling ────────────────────────────────────────────────────────

class TestBacktestRolling:
    def test_predicao_perfeita_zera_metricas(self):
        N, H, T = 4, 5, 2
        y = np.random.default_rng(0).normal(size=(N, H, T)).astype(np.float32)
        bt = backtest_rolling(y, y)
        assert bt.mae_global == pytest.approx(0.0)
        assert bt.rmse_global == pytest.approx(0.0)
        assert bt.janelas_avaliadas == N

    def test_metricas_por_horizonte_tem_tamanho_correto(self):
        N, H, T = 3, 7, 2
        rng = np.random.default_rng(1)
        pred = rng.normal(size=(N, H, T)).astype(np.float32)
        true = rng.normal(size=(N, H, T)).astype(np.float32)
        bt = backtest_rolling(pred, true)
        assert bt.mae_por_horizonte.shape == (H,)
        assert bt.rmse_por_horizonte.shape == (H,)

    def test_shapes_diferentes_falham(self):
        with pytest.raises(ValueError):
            backtest_rolling(np.zeros((3, 5, 2)), np.zeros((3, 4, 2)))


# ─── Resumo formatado ───────────────────────────────────────────────────────

class TestResumo:
    def test_resumo_contem_horizonte(self):
        N, H, T = 2, 3, 1
        pred = np.zeros((N, H, T))
        true = np.ones((N, H, T))
        bt = backtest_rolling(pred, true)
        s = bt.resumo()
        assert "MAE" in s
        assert "+1d" in s
        assert "+3d" in s
