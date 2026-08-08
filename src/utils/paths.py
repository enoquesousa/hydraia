"""
HydraIA Path Manager
====================
Centraliza todos os caminhos de arquivos e diretórios do projeto.
Facilita a migração de dados e mantém a raiz limpa.
"""

import os
from pathlib import Path

# Raiz do Projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Camada de Armazenamento (Storage)
STORAGE_DIR = BASE_DIR / "storage"

# Dados Brutos (Raw)
RAW_DIR = STORAGE_DIR / "raw"
SABESP_RAW_DIR = RAW_DIR / "sabesp"

# Dados Processados (Processed)
PROCESSED_DIR = STORAGE_DIR / "processed"
DATASETS_DIR = PROCESSED_DIR / "datasets"
HISTORY_DIR = PROCESSED_DIR / "history"

# Modelos (Models)
MODELS_STORE = STORAGE_DIR / "models"
LLM_MODELS_DIR = MODELS_STORE / "llm"
TS_MODELS_DIR = MODELS_STORE / "time_series"

# Saídas (Outputs)
OUTPUTS_DIR = STORAGE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
JSON_OUT_DIR = OUTPUTS_DIR / "json"

class HydraPaths:
    """Namespace para acesso simplificado aos caminhos."""
    BASE = BASE_DIR
    STORAGE = STORAGE_DIR

    # Raw
    SABESP = SABESP_RAW_DIR

    # Processed
    DATASETS = DATASETS_DIR
    HISTORY = HISTORY_DIR

    # Models
    LLM = LLM_MODELS_DIR
    TIME_SERIES = TS_MODELS_DIR

    # Outputs
    REPORTS = REPORTS_DIR
    JSON = JSON_OUT_DIR

    @staticmethod
    def ensure_dirs():
        """Garante que todos os diretórios necessários existam."""
        dirs = [
            SABESP_RAW_DIR,
            DATASETS_DIR, HISTORY_DIR,
            LLM_MODELS_DIR, TS_MODELS_DIR,
            REPORTS_DIR, JSON_OUT_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    HydraPaths.ensure_dirs()
    print("✅ Estrutura de diretórios de armazenamento verificada.")
