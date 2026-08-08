"""Motor de Inferência HydraIA - Séries Temporais.
Carrega o modelo salvo em storage/models/time_series/hydro_lstm_melhor.pt,
lê os últimos LOOKBACK dias de dados reais da SABESP e gera a previsão.

ESTRITAMENTE REAL: Se o modelo ou os dados não estiverem disponíveis, lança erro.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.data.baselines import prob_evento
from src.data.constantes import PARAMS, SISTEMAS
from src.data.modelo_lstm import HydroLSTM
from src.utils.paths import HydraPaths

DATASETS_DIR = HydraPaths.DATASETS
MODELS_DIR   = HydraPaths.TIME_SERIES
BASE_SERIES  = HydraPaths.SABESP / "series" / "sistemas"
RESIDUOS_CACHE = MODELS_DIR / "residuos_val.npy"
logger = logging.getLogger(__name__)


def _carregar_residuos_validacao(
    model: HydroLSTM,
    device: torch.device,
    n_targets: int,
    horizon: int,
    scaler_y,
    batch: int = 128,
) -> Optional[np.ndarray]:
    """Calcula (ou carrega cache) de resíduos no split de validação.

    Retorna array (N, horizon, n_targets) com (real - previsto) desnormalizado.
    """
    if RESIDUOS_CACHE.exists():
        try:
            data = np.load(RESIDUOS_CACHE)
            if data.ndim == 3 and data.shape[1] >= horizon and data.shape[2] == n_targets:
                return data
        except (OSError, ValueError) as exc:
            logger.warning("Cache de residuos invalido; sera recalculado: %s", exc)

    npz_path = DATASETS_DIR / "sabesp_dataset.npz"
    if not npz_path.exists():
        return None

    try:
        with np.load(npz_path) as npz:
            X_val = npz.get("X_val")
            y_val = npz.get("y_val")
            if X_val is not None:
                X_val = X_val.copy()
            if y_val is not None:
                y_val = y_val.copy()
    except (OSError, ValueError) as exc:
        logger.warning("Dataset de validacao indisponivel para incerteza: %s", exc)
        return None
    if X_val is None or y_val is None:
        return None

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch):
            X_b = torch.from_numpy(X_val[i : i + batch]).to(device)
            pred = model(X_b).cpu().numpy()
            preds.append(pred)
    if not preds:
        return None

    preds_arr = np.concatenate(preds, axis=0)
    y_arr = y_val

    # Desnormaliza
    pred_dn = scaler_y.inverse_transform(preds_arr.reshape(-1, n_targets)).reshape(
        preds_arr.shape[0], horizon, n_targets
    )
    y_dn = scaler_y.inverse_transform(y_arr.reshape(-1, n_targets)).reshape(
        y_arr.shape[0], horizon, n_targets
    )
    residuos = y_dn - pred_dn
    try:
        np.save(RESIDUOS_CACHE, residuos)
    except (OSError, ValueError) as exc:
        logger.warning("Nao foi possivel persistir cache de residuos: %s", exc)
    return residuos


def _intervalos_predicao(
    pred_h: np.ndarray,  # (horizon, n_targets)
    residuos: Optional[np.ndarray],  # (N, horizon, n_targets)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Calcula p10/p50/p90 e retorna resíduos por horizonte alvo (ou None)."""
    if residuos is None or residuos.size == 0:
        p10 = p50 = p90 = pred_h
        return p10, p50, p90, None

    q = np.quantile(residuos, [0.10, 0.50, 0.90], axis=0)  # (3, horizon, n_targets)
    p10 = pred_h + q[0]
    p50 = pred_h + q[1]
    p90 = pred_h + q[2]
    return p10, p50, p90, residuos

def _carregar_ultimos_dias(lookback: int) -> Optional[np.ndarray]:
    import pandas as pd
    series_list = []
    for el_id in SISTEMAS:
        el_dir = BASE_SERIES / el_id
        for param_arq in PARAMS:
            path = el_dir / f"{param_arq}.json"
            if not path.exists():
                raise FileNotFoundError(f"❌ Erro: Dado real ausente em {path}. Pipeline interrompido.")
            raw = json.loads(path.read_text(encoding="utf-8"))
            registros = raw.get("dados", [])
            if not registros:
                raise ValueError(f"❌ Erro: Série temporal vazia para {el_id}. Pipeline interrompido.")
            df_tmp = pd.DataFrame(registros)[["date", "value"]]
            df_tmp["date"] = pd.to_datetime(df_tmp["date"])
            df_tmp = df_tmp.set_index("date")["value"]
            df_tmp.index = df_tmp.index.normalize()
            df_tmp = df_tmp[~df_tmp.index.duplicated(keep="first")]
            series_list.append(df_tmp.rename(f"{el_id}__{param_arq}"))

    df = pd.concat(series_list, axis=1).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.interpolate(method="time", limit=7).ffill().bfill()

    if len(df) < lookback:
        raise ValueError(f"❌ Erro: Histórico insuficiente ({len(df)}/{lookback} dias). Coleta real incompleta.")

    janela = df.iloc[-lookback:].values.astype(np.float32)
    idx = df.index[-lookback:]
    doy = idx.day_of_year.values
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    month_n = (idx.month.values - 1) / 11.0
    temp_feat = np.stack([sin_doy, cos_doy, month_n], axis=1).astype(np.float32)
    return np.concatenate([janela, temp_feat], axis=1)

def prever(horizonte_dias: int = 30, *, otimizar: bool = True) -> dict:
    """Gera previsão de v_util (%) com intervalos e probabilidades.

    Args:
        horizonte_dias: dias a prever.
        otimizar: se True, tenta usar torch.compile e quantizacao dinamica.
    """
    if not (MODELS_DIR / "hydro_lstm_melhor.pt").exists():
        raise RuntimeError("❌ Erro Crítico: Modelo HydroLSTM não encontrado. Execute o treinamento com dados reais.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((MODELS_DIR / "hydro_lstm_meta.json").read_text(encoding="utf-8"))

    with open(DATASETS_DIR / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(DATASETS_DIR / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    lookback = meta["lookback"]
    n_features = meta["n_features"]
    n_targets = meta["n_targets"]
    horizon = meta["horizon"]
    sistemas_nomes = meta["sistemas"]

    janela_raw = _carregar_ultimos_dias(lookback)

    janela_hidro = janela_raw[:, :n_features - 3]
    janela_temp = janela_raw[:, n_features - 3:]
    janela_hidro_n = scaler_X.transform(janela_hidro)
    janela_n = np.concatenate([janela_hidro_n, janela_temp], axis=1).astype(np.float32)
    X = torch.from_numpy(janela_n).unsqueeze(0).to(device)

    model = HydroLSTM(n_features=n_features, n_targets=n_targets, horizon=horizon,
                      hidden=meta["hidden"], n_layers=meta["n_layers"], dropout=0.0)
    model.load_state_dict(torch.load(MODELS_DIR / "hydro_lstm_melhor.pt", map_location=device))
    model.to(device)
    model.eval()

    if otimizar and device.type == "cpu":
        try:
            # Quantizacao dinamica para CPU (muito rapida para LSTM)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Nao foi possivel quantizar o modelo: %s", e)

    if otimizar and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("torch.compile nao disponivel: %s", e)

    with torch.no_grad():
        pred_n = model(X).cpu().numpy()[0]

    pred = scaler_y.inverse_transform(pred_n)
    dias_a_usar = min(horizonte_dias, horizon)
    pred = pred[:dias_a_usar]

    # Resíduos de validação para incerteza
    residuos_val = _carregar_residuos_validacao(
        model=model,
        device=device,
        n_targets=n_targets,
        horizon=horizon,
        scaler_y=scaler_y,
    )
    if residuos_val is not None and residuos_val.shape[1] > dias_a_usar:
        residuos_val = residuos_val[:, :dias_a_usar, :]

    p10, p50, p90, residuos_h = _intervalos_predicao(pred, residuos_val)

    hoje = date.today()
    resultado = {}
    for i, sistema in enumerate(sistemas_nomes):
        if i >= n_targets: break
        dias = []
        for d in range(dias_a_usar):
            data_str = str(hoje + timedelta(days=d + 1))
            v = float(np.clip(pred[d, i], 0.0, 150.0))
            entry = {
                "data": data_str,
                "v_util_pct": round(v, 2),
                "p10": round(float(np.clip(p10[d, i], 0.0, 150.0)), 2),
                "p50": round(float(np.clip(p50[d, i], 0.0, 150.0)), 2),
                "p90": round(float(np.clip(p90[d, i], 0.0, 150.0)), 2),
            }

            if residuos_h is not None:
                res_vec = residuos_h[:, d, i]
                entry["prob_sub_30_pct"] = round(prob_evento(v, res_vec, 30.0, direcao="abaixo") * 100, 2)
                entry["prob_sub_50_pct"] = round(prob_evento(v, res_vec, 50.0, direcao="abaixo") * 100, 2)
                entry["prob_acima_80_pct"] = round(prob_evento(v, res_vec, 80.0, direcao="acima") * 100, 2)
            dias.append(entry)
        resultado[sistema] = dias

    return resultado
