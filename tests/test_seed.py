"""Testes da seed global: reprodutibilidade entre runs."""

from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import seed_global


class TestReprodutibilidade:
    def test_random_python_reproduzivel(self):
        seed_global(123)
        a = [random.random() for _ in range(5)]
        seed_global(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_reproduzivel(self):
        seed_global(7)
        a = np.random.rand(10).tolist()
        seed_global(7)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_seeds_diferentes_geram_sequencias_diferentes(self):
        seed_global(1)
        a = [random.random() for _ in range(3)]
        seed_global(2)
        b = [random.random() for _ in range(3)]
        assert a != b

    def test_pythonhashseed_e_setado(self):
        seed_global(99)
        assert os.environ["PYTHONHASHSEED"] == "99"
