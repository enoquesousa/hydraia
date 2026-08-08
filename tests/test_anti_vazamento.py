"""Testa que a imputacao por split nao vaza informacao do futuro."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.imputacao import imputar_split


class TestImputarSplit:
    def test_imputa_dentro_do_split(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=idx)
        out = imputar_split(df)
        assert out.isna().sum().sum() == 0
        assert out.iloc[1, 0] == pytest.approx(2.0)
        assert out.iloc[3, 0] == pytest.approx(4.0)

    def test_seed_ffill_ancora_inicio_do_split(self):
        # split anterior: termina em 99 (passado, OK usar como ancora)
        idx_a = pd.date_range("2020-01-01", periods=5, freq="D")
        df_a = pd.DataFrame({"x": [95, 96, 97, 98, 99]}, index=idx_a, dtype=float)
        # split atual: comeca com NaN
        idx_b = pd.date_range("2020-01-06", periods=5, freq="D")
        df_b = pd.DataFrame({"x": [np.nan, np.nan, 50, 60, 70]}, index=idx_b, dtype=float)
        out = imputar_split(df_b, seed_ffill=df_a)
        # primeiro valor nao deve ser NaN (foi imputado a partir do passado)
        assert not pd.isna(out.iloc[0, 0])
        # nao introduz vazamento porque a ancora e do PASSADO
        assert out.iloc[2, 0] == pytest.approx(50.0)
        assert out.iloc[4, 0] == pytest.approx(70.0)

    def test_imputacao_so_dentro_do_split_quando_sem_seed(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame({"x": [np.nan, np.nan, 10.0, 20.0]}, index=idx)
        out = imputar_split(df, seed_ffill=None)
        # bfill faz primeiros virarem 10
        assert out.iloc[0, 0] == 10.0
        assert out.iloc[1, 0] == 10.0

    def test_split_atual_nao_polui_split_anterior(self):
        """A imputacao do split atual usa o anterior como ancora, mas nao
        retorna o anterior — so o pedaco do split atual."""
        idx_a = pd.date_range("2020-01-01", periods=3, freq="D")
        df_a = pd.DataFrame({"x": [1, 2, 3]}, index=idx_a, dtype=float)
        idx_b = pd.date_range("2020-01-04", periods=3, freq="D")
        df_b = pd.DataFrame({"x": [4, 5, 6]}, index=idx_b, dtype=float)
        out = imputar_split(df_b, seed_ffill=df_a)
        # so retorna as 3 linhas do split atual
        assert len(out) == 3
        assert list(out.index) == list(idx_b)
