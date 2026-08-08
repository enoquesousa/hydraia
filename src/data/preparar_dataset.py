"""Prepara dataset de series temporais dos sistemas SABESP para treinamento.

Le dados JSON de dados_sabesp/series/sistemas/, cria DataFrame unificado
com todos os parametros hidrologicos, normaliza e salva janelas deslizantes
em datasets/sabesp_dataset.npz para uso nos modelos de previsao.

Sistemas usados (dados completos 2000-2026):
  Cantareira, Alto Tiete, Guarapiranga, Cotia, Rio Grande, Rio Claro

Parametros por sistema:
  chuva (mm), v_util (%), q_natural (m3/s), q_afluente (m3/s),
  q_captada (m3/s), q_jusante (m3/s)

Targets (o que o modelo aprende a prever):
  v_util (%) para cada um dos 6 sistemas — horizonte 30 dias

CORRECOES (Fase 3 — anti-vazamento temporal):
  * Imputacao por interpolacao SO no split de treino — val/test usa ffill/bfill
    a partir do split anterior (sem olhar pro futuro).
  * Seed global fixa para reprodutibilidade (`HYDRAIA_SEED`, default 42).
  * Salva `media_por_mes` e `target_idx_in_features` para baselines.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.constantes import (
    DATA_TREINO_FIM,
    DATA_VAL_FIM,
    HORIZON,
    LOOKBACK,
    PARAMS,
    SISTEMAS,
    TARGET_PARAM,
)
from src.data.imputacao import imputar_split as _imputar_split
from src.utils.paths import HydraPaths
from src.utils.seed import seed_global

BASE_SERIES = HydraPaths.SABESP / "series/sistemas"
SAIDA_DIR = HydraPaths.DATASETS


# ─── carregamento ─────────────────────────────────────────────────────────────

def _carregar_serie(el_dir: Path, param_arquivo: str) -> pd.Series:
    """Le um arquivo JSON de serie temporal e retorna pd.Series indexada por data."""
    path = el_dir / f"{param_arquivo}.json"
    if not path.exists():
        return pd.Series(dtype=float)
    raw = json.loads(path.read_text(encoding="utf-8"))
    registros = raw.get("dados", [])
    if not registros:
        return pd.Series(dtype=float)
    df = pd.DataFrame(registros)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")["value"]
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


def construir_dataframe_bruto() -> pd.DataFrame:
    """Monta DataFrame multi-coluna (sistema, parametro) — SEM imputacao.

    A imputacao acontece DEPOIS, ja dentro de cada split, para nao vazar
    informacao do futuro para o passado.
    """
    print("[dataset] Carregando series da API SABESP (bruto, sem imputacao)...")
    series: dict[tuple[str, str], pd.Series] = {}

    for el_id, el_nome in SISTEMAS.items():
        el_dir = BASE_SERIES / el_id
        for param_arq, param_nome in PARAMS.items():
            s = _carregar_serie(el_dir, param_arq)
            if s.empty:
                print(f"  AVISO: {el_nome}/{param_nome} vazio, pulando.")
                continue
            series[(el_nome, param_nome)] = s
            print(f"  {el_nome}/{param_nome}: {len(s)} registros "
                  f"({s.index.min().date()} a {s.index.max().date()})")

    df = pd.DataFrame(series)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["sistema", "parametro"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    print(f"\n[dataset] DataFrame bruto: {df.shape[0]} dias x {df.shape[1]} colunas")
    print(f"          Periodo: {df.index.min().date()} a {df.index.max().date()}")
    print(f"          NaNs antes de imputar: {df.isna().sum().sum()}")
    return df


# ─── features temporais ───────────────────────────────────────────────────────

def _adicionar_features_temporais(idx: pd.DatetimeIndex) -> np.ndarray:
    """Adiciona seno/cosseno do dia do ano e mes normalizado como colunas extras."""
    doy = idx.day_of_year.values
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    month_norm = (idx.month.values - 1) / 11.0
    return np.stack([sin_doy, cos_doy, month_norm], axis=1)


# ─── criacao de janelas ───────────────────────────────────────────────────────

def _criar_janelas(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cria arrays de janelas deslizantes (X, y)."""
    Xs, ys = [], []
    total = len(X_arr)
    for i in range(lookback, total - horizon + 1):
        Xs.append(X_arr[i - lookback : i])
        ys.append(y_arr[i : i + horizon])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─── pipeline principal ───────────────────────────────────────────────────────

def preparar(
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    seed: int | None = None,
) -> None:
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    seed_efetiva = seed if seed is not None else int(os.environ.get("HYDRAIA_SEED", "42"))
    seed_global(seed_efetiva)
    print(f"[dataset] Seed global = {seed_efetiva}")

    df = construir_dataframe_bruto()

    feat_cols = list(df.columns)
    tgt_cols = [(s, TARGET_PARAM) for s in SISTEMAS.values() if (s, TARGET_PARAM) in df.columns]

    features_df = df[feat_cols].copy()
    targets_df = df[tgt_cols].copy()

    # ── splits por data (anti-vazamento: imputar DENTRO de cada split) ──
    idx_treino_fim = df.index.searchsorted(DATA_TREINO_FIM, side="right")
    idx_val_fim = df.index.searchsorted(DATA_VAL_FIM, side="right")

    feat_treino_df = features_df.iloc[:idx_treino_fim]
    feat_val_df = features_df.iloc[idx_treino_fim:idx_val_fim]
    feat_teste_df = features_df.iloc[idx_val_fim:]

    feat_treino_df = _imputar_split(feat_treino_df)
    feat_val_df = _imputar_split(feat_val_df, seed_ffill=feat_treino_df)
    feat_teste_df = _imputar_split(feat_teste_df, seed_ffill=feat_val_df)

    tgt_treino_df = targets_df.iloc[:idx_treino_fim]
    tgt_val_df = targets_df.iloc[idx_treino_fim:idx_val_fim]
    tgt_teste_df = targets_df.iloc[idx_val_fim:]
    tgt_treino_df = _imputar_split(tgt_treino_df)
    tgt_val_df = _imputar_split(tgt_val_df, seed_ffill=tgt_treino_df)
    tgt_teste_df = _imputar_split(tgt_teste_df, seed_ffill=tgt_val_df)

    feat_treino = feat_treino_df.values
    feat_val = feat_val_df.values
    feat_teste = feat_teste_df.values
    tgt_treino = tgt_treino_df.values
    tgt_val = tgt_val_df.values
    tgt_teste = tgt_teste_df.values

    # ── normalizacao (fit apenas no treino) ──
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    feat_treino_n = scaler_X.fit_transform(feat_treino)
    feat_val_n = scaler_X.transform(feat_val)
    feat_teste_n = scaler_X.transform(feat_teste)

    tgt_treino_n = scaler_y.fit_transform(tgt_treino)
    tgt_val_n = scaler_y.transform(tgt_val)
    tgt_teste_n = scaler_y.transform(tgt_teste)

    # ── adicionar features temporais (apos normalizacao) ──
    feat_treino_n = np.concatenate(
        [feat_treino_n, _adicionar_features_temporais(feat_treino_df.index)], axis=1,
    )
    feat_val_n = np.concatenate(
        [feat_val_n, _adicionar_features_temporais(feat_val_df.index)], axis=1,
    )
    feat_teste_n = np.concatenate(
        [feat_teste_n, _adicionar_features_temporais(feat_teste_df.index)], axis=1,
    )

    # ── janelas deslizantes ──
    print(f"\n[dataset] Criando janelas: lookback={lookback} dias, horizon={horizon} dias...")
    X_tr, y_tr = _criar_janelas(feat_treino_n, tgt_treino_n, lookback, horizon)
    X_va, y_va = _criar_janelas(feat_val_n, tgt_val_n, lookback, horizon)
    X_te, y_te = _criar_janelas(feat_teste_n, tgt_teste_n, lookback, horizon)
    print(f"  Treino : X{X_tr.shape} y{y_tr.shape}")
    print(f"  Val    : X{X_va.shape} y{y_va.shape}")
    print(f"  Teste  : X{X_te.shape} y{y_te.shape}")

    # ── baseline support: media por mes (climatologia) sobre o TREINO ──
    media_mes = (
        tgt_treino_df.assign(mes=tgt_treino_df.index.month)
        .groupby("mes").mean(numeric_only=True).values  # (12, n_targets)
    )

    # indice das colunas-alvo dentro de feat_cols (sem features temporais)
    target_idx_in_features = [feat_cols.index(c) for c in tgt_cols]

    # ── salvar ──
    np.savez_compressed(
        SAIDA_DIR / "sabesp_dataset.npz",
        X_treino=X_tr, y_treino=y_tr,
        X_val=X_va, y_val=y_va,
        X_teste=X_te, y_teste=y_te,
        media_mes=np.asarray(media_mes, dtype=np.float32),
    )

    with open(SAIDA_DIR / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(SAIDA_DIR / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    meta = {
        "lookback": lookback,
        "horizon": horizon,
        "n_features": feat_treino_n.shape[1],
        "n_targets": tgt_treino_n.shape[1],
        "feature_cols": [f"{s}__{p}" for s, p in feat_cols],
        "target_cols": [f"{s}__{p}" for s, p in tgt_cols],
        "target_idx_in_features": target_idx_in_features,
        "sistemas": list(SISTEMAS.values()),
        "params": list(PARAMS.values()),
        "target_param": TARGET_PARAM,
        "treino_fim": DATA_TREINO_FIM,
        "val_fim": DATA_VAL_FIM,
        "data_inicio": str(df.index.min().date()),
        "data_fim": str(df.index.max().date()),
        "seed": seed_efetiva,
        "anti_vazamento_temporal": True,
    }
    with open(SAIDA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[dataset] Salvo em {SAIDA_DIR}/")
    print(f"  sabesp_dataset.npz — scaler_X.pkl — scaler_y.pkl — meta.json")
    print(f"  Anti-vazamento temporal: ativado")
    print(f"  Seed global: {seed_efetiva}")


if __name__ == "__main__":
    preparar()
