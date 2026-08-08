"""Reprodutibilidade — seed unica para Python, NumPy e PyTorch."""

from __future__ import annotations

import os
import random


def seed_global(seed: int = 42) -> None:
    """Fixa seed em todas as fontes de aleatoriedade que importam para o pipeline.

    Importa torch tardiamente para nao depender da instalacao em ambientes
    sem GPU. Tambem fixa a flag determinista do CuDNN, com o trade-off de
    tornar treinos um pouco mais lentos em troca de runs reproduziveis.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinismo: trade-off de velocidade
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
