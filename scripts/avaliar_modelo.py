"""Reavalia um checkpoint HydroLSTM e emite métricas com proveniência.

Uso:
    python scripts/avaliar_modelo.py
    python scripts/avaliar_modelo.py --output artifacts/model_evaluation.local.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.modelo_lstm import HydroLSTM  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metricas(
    model: HydroLSTM,
    X: np.ndarray,
    y: np.ndarray,
    scaler_y,
    *,
    batch_size: int,
) -> dict[str, float | int]:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).float()
            preds.append(model(xb).cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    n, horizon, targets = pred.shape
    pred_pp = scaler_y.inverse_transform(pred.reshape(-1, targets)).reshape(
        n, horizon, targets
    )
    y_pp = scaler_y.inverse_transform(y.reshape(-1, targets)).reshape(
        n, horizon, targets
    )
    return {
        "janelas": int(n),
        "mae_normalizado": float(np.mean(np.abs(pred - y))),
        "mae_pp": float(np.mean(np.abs(pred_pp - y_pp))),
        "rmse_pp": float(np.sqrt(np.mean((pred_pp - y_pp) ** 2))),
    }


def avaliar(root: Path = ROOT, *, batch_size: int = 128) -> dict:
    dataset_path = root / "storage/processed/datasets/sabesp_dataset.npz"
    scaler_y_path = root / "storage/processed/datasets/scaler_y.pkl"
    checkpoint_path = root / "storage/models/time_series/hydro_lstm_melhor.pt"
    metadata_path = root / "storage/models/time_series/hydro_lstm_meta.json"

    paths = [dataset_path, scaler_y_path, checkpoint_path, metadata_path]
    ausentes = [str(path.relative_to(root)) for path in paths if not path.exists()]
    if ausentes:
        raise FileNotFoundError("Artefatos ausentes: " + ", ".join(ausentes))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with scaler_y_path.open("rb") as stream:
        scaler_y = pickle.load(stream)

    model = HydroLSTM(
        n_features=metadata["n_features"],
        n_targets=metadata["n_targets"],
        horizon=metadata["horizon"],
        hidden=metadata["hidden"],
        n_layers=metadata["n_layers"],
        dropout=0.0,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with np.load(dataset_path) as dataset:
        validacao = _metricas(
            model,
            dataset["X_val"],
            dataset["y_val"],
            scaler_y,
            batch_size=batch_size,
        )
        teste = _metricas(
            model,
            dataset["X_teste"],
            dataset["y_teste"],
            scaler_y,
            batch_size=batch_size,
        )

    return {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "dataset_sha256": _sha256(dataset_path),
            "scaler_y_sha256": _sha256(scaler_y_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "metadata_sha256": _sha256(metadata_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "device": "cpu",
        },
        "results": {"validation": validacao, "test": teste},
        "interpretation": {
            "mae_normalizado": "adimensional; usado no early stopping",
            "mae_pp": "erro absoluto medio apos inverse_transform, em pontos percentuais",
            "rmse_pp": "raiz do erro quadratico medio apos inverse_transform, em pontos percentuais",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="salva o JSON além de imprimi-lo")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    try:
        result = avaliar(batch_size=args.batch_size)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"Erro de avaliação: {exc}", file=sys.stderr)
        return 2

    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
