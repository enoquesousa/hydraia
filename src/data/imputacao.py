"""Imputacao temporal SEM vazamento — usado por preparar_dataset.

Modulo isolado para nao depender de sklearn (e portanto testavel sem ele).
"""

from __future__ import annotations

import pandas as pd


def imputar_split(df_split: pd.DataFrame, seed_ffill: pd.DataFrame | None = None) -> pd.DataFrame:
    """Imputa um split sem olhar para o futuro alem dele.

    1. Interpolacao temporal limitada a 7 dias DENTRO do split.
    2. Se o split comeca com NaNs e ha um split anterior (`seed_ffill`),
       usa as ultimas 30 linhas dele como ancora — sem revelar o futuro
       do split atual.
    3. ffill/bfill final SO dentro do split.
    """
    out = df_split.copy()
    if seed_ffill is not None and not seed_ffill.empty:
        cauda = seed_ffill.tail(30)
        ext = pd.concat([cauda, out])
        ext = ext.interpolate(method="time", limit=7).ffill().bfill()
        out = ext.loc[out.index]
    else:
        out = out.interpolate(method="time", limit=7).ffill().bfill()
    return out
