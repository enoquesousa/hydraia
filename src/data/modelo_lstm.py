"""Arquitetura HydroLSTM — definição única usada por treino e inferência.

Single Source of Truth: alterar aqui e o checkpoint salvo no treino continua
sendo carregável por `inferencia_ts.py` sem dessincronia.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HydroLSTM(nn.Module):
    """LSTM bidirecional para previsão multivariada de reservatórios.

    Args:
        n_features: número de features de entrada (parâmetros hidrológicos +
            features temporais sin/cos do dia do ano + mês normalizado).
        n_targets: número de séries a prever simultaneamente (um v_util por sistema).
        horizon: número de passos à frente que a cabeça projeta.
        hidden: tamanho do estado oculto da LSTM por direção.
        n_layers: número de camadas empilhadas da LSTM.
        dropout: taxa de dropout entre camadas (ignorada se n_layers == 1).
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        horizon: int,
        hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, horizon * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        last = self.drop(out[:, -1, :])
        pred = self.head(last)
        return pred.view(-1, self.horizon, self.n_targets)
