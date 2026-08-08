"""Treina LSTM multivariado nos dados historicos SABESP para prever v_util (%).

Arquitetura:
  - 2 camadas LSTM bidirecionais com dropout
  - Cabeca linear para projetar para (horizon * n_targets)
  - Loss: MAE + MSE ponderados (robusto a outliers hidrologicos)
  - Early stopping baseado em MAE de validacao

CORRECOES (Fase 3):
  * Seed global fixa (`HYDRAIA_SEED`, default 42).
  * Avaliacao no teste inclui baselines (persistence, drift, sazonal) e
    backtesting rolling — gravados em hydro_lstm_meta.json.

Antes de rodar, execute preparar_dataset.py para gerar datasets/sabesp_dataset.npz.

Uso:
    python -m src.data.treinar_modelo_ts
    python -m src.data.treinar_modelo_ts --epochs 100 --batch 64 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.baselines import backtest_rolling, comparar_baselines
from src.data.modelo_lstm import HydroLSTM
from src.utils.paths import HydraPaths
from src.utils.seed import seed_global

DATASETS_DIR = HydraPaths.DATASETS
MODELS_DIR = HydraPaths.TIME_SERIES


# ─── loss ─────────────────────────────────────────────────────────────────────

class HydroLoss(nn.Module):
    """MAE + MSE ponderados — equilibra outliers e suavidade."""

    def __init__(self, alpha: float = 0.6) -> None:
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ─── treinamento ──────────────────────────────────────────────────────────────

def _carregar_datasets(
    datasets_dir: Path,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict, np.ndarray]:
    npz = np.load(datasets_dir / "sabesp_dataset.npz")
    meta = json.loads((datasets_dir / "meta.json").read_text(encoding="utf-8"))

    def _t(key: str) -> torch.Tensor:
        return torch.from_numpy(npz[key].astype(np.float32))

    media_mes = npz["media_mes"] if "media_mes" in npz.files else None
    ds_tr = TensorDataset(_t("X_treino"), _t("y_treino"))
    ds_va = TensorDataset(_t("X_val"), _t("y_val"))
    ds_te = TensorDataset(_t("X_teste"), _t("y_teste"))
    return ds_tr, ds_va, ds_te, meta, media_mes


def treinar(
    epochs: int = 150,
    batch: int = 128,
    lr: float = 1e-3,
    hidden: int = 256,
    n_layers: int = 2,
    dropout: float = 0.25,
    patience: int = 20,
    seed: int | None = None,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seed_efetiva = seed if seed is not None else int(os.environ.get("HYDRAIA_SEED", "42"))
    seed_global(seed_efetiva)
    print(f"[treino] Seed global = {seed_efetiva}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[treino] Dispositivo: {device}")

    print("[treino] Carregando dataset...")
    ds_tr, ds_va, ds_te, meta, media_mes = _carregar_datasets(DATASETS_DIR)
    n_features = meta["n_features"]
    n_targets = meta["n_targets"]
    horizon = meta["horizon"]

    # Loader de treino: shuffle COM gerador determinista
    g_tr = torch.Generator()
    g_tr.manual_seed(seed_efetiva)
    loader_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,
                           drop_last=True, generator=g_tr)
    loader_va = DataLoader(ds_va, batch_size=batch, shuffle=False)
    loader_te = DataLoader(ds_te, batch_size=batch, shuffle=False)

    print(f"  n_features={n_features}  n_targets={n_targets}  horizon={horizon}")
    print(f"  Janelas — treino:{len(ds_tr)} val:{len(ds_va)} teste:{len(ds_te)}")

    model = HydroLSTM(
        n_features=n_features,
        n_targets=n_targets,
        horizon=horizon,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros do modelo: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = HydroLoss(alpha=0.6).to(device)

    melhor_val_mae = float("inf")
    epochs_sem_melhora = 0
    historico: list[dict] = []

    print(f"\n[treino] Iniciando ({epochs} epocas, early stopping patience={patience})...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_tr = 0.0
        for X_b, y_b in loader_tr:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_tr += loss.item() * len(X_b)
        loss_tr /= len(ds_tr)

        model.eval()
        mae_va_list = []
        with torch.no_grad():
            for X_b, y_b in loader_va:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                mae = nn.functional.l1_loss(pred, y_b, reduction="mean")
                mae_va_list.append(mae.item())
        mae_va = float(np.mean(mae_va_list))

        scheduler.step()
        historico.append({"epoch": epoch, "loss_treino": round(loss_tr, 5), "mae_val": round(mae_va, 5)})

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoca {epoch:4d}/{epochs}  loss_treino={loss_tr:.4f}  mae_val={mae_va:.4f}  ({elapsed:.0f}s)")

        if mae_va < melhor_val_mae - 1e-5:
            melhor_val_mae = mae_va
            epochs_sem_melhora = 0
            torch.save(model.state_dict(), MODELS_DIR / "hydro_lstm_melhor.pt")
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                print(f"\n  Early stopping na epoca {epoch} (sem melhora ha {patience} epocas)")
                break

    # ── avaliacao no teste (LSTM + baselines + backtest rolling) ──
    model.load_state_dict(torch.load(MODELS_DIR / "hydro_lstm_melhor.pt", map_location=device))
    model.eval()

    with open(DATASETS_DIR / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    with open(DATASETS_DIR / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    def _inferir(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds_all, targets_all, X_all = [], [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                pred = model(X_b.to(device)).cpu().numpy()
                preds_all.append(pred)
                targets_all.append(y_b.numpy())
                X_all.append(X_b.numpy())
        return (
            np.concatenate(preds_all, axis=0),
            np.concatenate(targets_all, axis=0),
            np.concatenate(X_all, axis=0),
        )

    preds_val, targets_val, _ = _inferir(loader_va)
    preds, targets, X_te_arr = _inferir(loader_te)

    N, H, T = preds.shape
    preds_dn = scaler_y.inverse_transform(preds.reshape(-1, T)).reshape(N, H, T)
    targets_dn = scaler_y.inverse_transform(targets.reshape(-1, T)).reshape(N, H, T)

    Nv, Hv, Tv = preds_val.shape
    preds_val_dn = scaler_y.inverse_transform(preds_val.reshape(-1, Tv)).reshape(Nv, Hv, Tv)
    targets_val_dn = scaler_y.inverse_transform(targets_val.reshape(-1, Tv)).reshape(Nv, Hv, Tv)
    mae_val_pp = float(np.mean(np.abs(preds_val_dn - targets_val_dn)))
    rmse_val_pp = float(np.sqrt(np.mean((preds_val_dn - targets_val_dn) ** 2)))

    # Desnormaliza tambem as features hidro do X (sem temporais) para baselines
    n_temp = 3
    X_hidro = X_te_arr[:, :, : n_features - n_temp]
    X_hidro_dn = scaler_X.inverse_transform(X_hidro.reshape(-1, n_features - n_temp)).reshape(
        N, X_te_arr.shape[1], n_features - n_temp
    )

    mae_teste = float(np.mean(np.abs(preds_dn - targets_dn)))
    rmse_teste = float(np.sqrt(np.mean((preds_dn - targets_dn) ** 2)))

    print(f"\n[treino] ── Resultado HydroLSTM (desnormalizado) ──")
    print(f"  VALIDACAO: MAE={mae_val_pp:.3f} pp | RMSE={rmse_val_pp:.3f} pp")
    print("  TESTE:")
    print(f"  MAE  = {mae_teste:.3f} %")
    print(f"  RMSE = {rmse_teste:.3f} %")

    # Baselines
    target_idx = list(meta.get("target_idx_in_features", []))
    baselines_metrics = {}
    if target_idx:
        bs = comparar_baselines(
            janelas_X=X_hidro_dn,
            y_true=targets_dn,
            target_idx_in_features=target_idx,
            media_por_mes=media_mes,
            mes_inicial_por_amostra=None,  # nao temos os meses por amostra aqui
        )
        print("\n[treino] ── Baselines no TESTE ──")
        for nome, m in bs.items():
            print(f"  {m}")
            baselines_metrics[nome] = {"mae": m.mae, "rmse": m.rmse}

    # Backtest rolling
    bt = backtest_rolling(preds_dn, targets_dn)
    print("\n[treino] ── Backtest rolling (LSTM) ──")
    print(bt.resumo())

    # Por sistema
    target_cols = meta["target_cols"]
    metricas_por_sistema = []
    for i, col in enumerate(target_cols):
        mae_i = float(np.mean(np.abs(preds_dn[:, :, i] - targets_dn[:, :, i])))
        rmse_i = float(np.sqrt(np.mean((preds_dn[:, :, i] - targets_dn[:, :, i]) ** 2)))
        print(f"    {col:<45} MAE={mae_i:.2f}%  RMSE={rmse_i:.2f}%")
        metricas_por_sistema.append({"sistema": col, "mae": mae_i, "rmse": rmse_i})

    modelo_meta = {
        "arquitetura": "HydroLSTM",
        "n_features": n_features,
        "n_targets": n_targets,
        "horizon": horizon,
        "hidden": hidden,
        "n_layers": n_layers,
        "dropout": dropout,
        "target_cols": target_cols,
        "sistemas": meta["sistemas"],
        "target_param": meta["target_param"],
        "mae_val_normalizado": round(melhor_val_mae, 4),
        "mae_val_pp": round(mae_val_pp, 4),
        "rmse_val_pp": round(rmse_val_pp, 4),
        "mae_teste": round(mae_teste, 4),
        "rmse_teste": round(rmse_teste, 4),
        "lookback": meta["lookback"],
        "seed": seed_efetiva,
        "baselines_teste": baselines_metrics,
        "metricas_por_sistema": metricas_por_sistema,
        "backtest_mae_por_horizonte": [float(x) for x in bt.mae_por_horizonte],
        "backtest_rmse_por_horizonte": [float(x) for x in bt.rmse_por_horizonte],
    }
    with open(MODELS_DIR / "hydro_lstm_meta.json", "w", encoding="utf-8") as f:
        json.dump(modelo_meta, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t0
    print(f"\n[treino] Concluido em {total_time/60:.1f} min")
    print(f"  Modelo salvo em: {MODELS_DIR}/hydro_lstm_melhor.pt")
    print(f"  Metadados em:    {MODELS_DIR}/hydro_lstm_meta.json")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina LSTM nos dados SABESP")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    treinar(
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        hidden=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
    )
