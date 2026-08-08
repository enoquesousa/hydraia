"""
Gerenciador de Persistência Hídrica (SQLite - BI Edition)
========================================================
Modelo dimensional otimizado para BI, Dashboards e integração com Front-end TS.
Utiliza chaves estrangeiras, índices e tabelas de dimensões para garantir
integridade e performance de consulta.
"""

from __future__ import annotations

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.paths import HydraPaths

logger = logging.getLogger(__name__)

DB_PATH = HydraPaths.BASE / "storage" / "hydraia_bi.db"

class PersistenceManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Inicializa o esquema dimensional (Star Schema)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()

            # 1. Dimensões
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dim_sistemas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT UNIQUE,
                    nome TEXT,
                    bacia TEXT,
                    capacidade_total_hm3 REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dim_tempo (
                    data TEXT PRIMARY KEY,
                    ano INTEGER,
                    mes INTEGER,
                    dia INTEGER,
                    doy INTEGER,
                    estacao TEXT
                )
            """)

            # 2. Fatos e Eventos (Relatórios)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fato_execucoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    correlation_id TEXT UNIQUE,
                    timestamp TEXT,
                    data_referencia TEXT,
                    risco_consolidado INTEGER,
                    confianca_ensemble_pct INTEGER,
                    relatorio_md TEXT,
                    metadados_json TEXT,
                    versao_codigo TEXT,
                    FOREIGN KEY (data_referencia) REFERENCES dim_tempo(data)
                )
            """)

            # 3. Medições Reais (Time Series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fato_medicoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execucao_id INTEGER,
                    sistema_id INTEGER,
                    data TEXT,
                    volume_pct REAL,
                    volume_hm3 REAL,
                    chuva_mm REAL,
                    vazao_natural REAL,
                    vazao_jusante REAL,
                    anomalia_zscore REAL,
                    delta_p50 REAL,
                    FOREIGN KEY (execucao_id) REFERENCES fato_execucoes(id),
                    FOREIGN KEY (sistema_id) REFERENCES dim_sistemas(id),
                    FOREIGN KEY (data) REFERENCES dim_tempo(data)
                )
            """)

            # 4. Previsões ML (Probabilístico)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fato_previsoes_ml (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execucao_id INTEGER,
                    sistema_id INTEGER,
                    data_previsao TEXT,
                    horizonte_dias INTEGER,
                    p10 REAL,
                    p50 REAL,
                    p90 REAL,
                    prob_sub_30 REAL,
                    prob_sub_50 REAL,
                    prob_acima_80 REAL,
                    FOREIGN KEY (execucao_id) REFERENCES fato_execucoes(id),
                    FOREIGN KEY (sistema_id) REFERENCES dim_sistemas(id),
                    FOREIGN KEY (data_previsao) REFERENCES dim_tempo(data)
                )
            """)

            # 5. Previsões Meteo
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fato_previsoes_meteo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execucao_id INTEGER,
                    data_previsao TEXT,
                    chuva_mm REAL,
                    temp_c REAL,
                    umidade_pct REAL,
                    descricao TEXT,
                    FOREIGN KEY (execucao_id) REFERENCES fato_execucoes(id)
                )
            """)

            # Índices para performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_medicoes_data ON fato_medicoes(data)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_ml_exec ON fato_previsoes_ml(execucao_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_previsoes_ml_data ON fato_previsoes_ml(data_previsao)")

            conn.commit()
            self._popular_dimensoes(conn)

    def _popular_dimensoes(self, conn):
        """Popula tabelas de dimensões estáticas."""
        from src.api.sabesp import SISTEMAS_SABESP
        cursor = conn.cursor()
        for slug, nome in SISTEMAS_SABESP.items():
            cursor.execute("INSERT OR IGNORE INTO dim_sistemas (slug, nome) VALUES (?, ?)", (slug, nome))
        conn.commit()

    def _get_sistema_id(self, cursor, slug: str) -> int:
        cursor.execute("SELECT id FROM dim_sistemas WHERE slug = ?", (slug,))
        res = cursor.fetchone()
        return res[0] if res else None

    def _garantir_tempo(self, cursor, data_iso: str):
        """Garante que a data existe na dimensão tempo."""
        dt = datetime.fromisoformat(data_iso[:10])
        estacao = "SECO" if 7 <= dt.month <= 9 else "CHUVOSO" if dt.month in [12, 1, 2] else "TRANSIÇÃO"
        cursor.execute("""
            INSERT OR IGNORE INTO dim_tempo (data, ano, mes, dia, doy, estacao)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data_iso[:10], dt.year, dt.month, dt.day, dt.timetuple().tm_yday, estacao))

    def salvar_execucao(
        self,
        relatorio: str,
        metadados: Dict[str, Any],
        estado_atual: Dict[str, Any],
        previsoes_ml: List[Dict[str, Any]],
        weather: Optional[Dict[str, Any]]
    ) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()

                data_ref = estado_atual.get("data_referencia", datetime.now().strftime("%Y-%m-%d"))
                self._garantir_tempo(cursor, data_ref)

                # 1. Fato Execução
                ensemble = metadados.get("ensemble", {})
                cursor.execute("""
                    INSERT INTO fato_execucoes (
                        correlation_id, timestamp, data_referencia,
                        risco_consolidado, confianca_ensemble_pct,
                        relatorio_md, metadados_json, versao_codigo
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadados.get("correlation_id"),
                    metadados.get("timestamp"),
                    data_ref,
                    ensemble.get("risco_consolidado"),
                    ensemble.get("confianca_ensemble_pct"),
                    relatorio,
                    json.dumps(metadados),
                    metadados.get("versao_codigo")
                ))
                exec_id = cursor.lastrowid

                # 2. Fato Medições (Snapshot Atual)
                sistemas = estado_atual.get("sistemas", {})
                anomalias = estado_atual.get("anomalias_historicas", {})
                from src.data.estatisticas_prob import _SABESP_TO_CANONICO

                for slug, info in sistemas.items():
                    sis_id = self._get_sistema_id(cursor, slug)
                    if not sis_id: continue

                    nome_canonico = _SABESP_TO_CANONICO.get(slug, slug)
                    anom = anomalias.get(nome_canonico, {})

                    cursor.execute("""
                        INSERT INTO fato_medicoes (
                            execucao_id, sistema_id, data, volume_pct, volume_hm3,
                            chuva_mm, vazao_natural, vazao_jusante, anomalia_zscore, delta_p50
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exec_id, sis_id, data_ref,
                        info.get("volume_percentual"),
                        info.get("volume_hm3"),
                        info.get("chuva_mm"),
                        info.get("vazao_natural_m3s"),
                        info.get("vazao_jusante_m3s"),
                        anom.get("zscore"),
                        (info.get("volume_percentual") - anom.get("p50_mes", 0)) if anom.get("p50_mes") else None
                    ))

                # 3. Fato Previsões ML
                for p in previsoes_ml:
                    # No momento o LSTM é fixo no Cantareira, mas o DB já aceita outros
                    # O contrato persistido atual usa Cantareira como serie principal.
                    slug_ml = "cantareira"
                    sis_id = self._get_sistema_id(cursor, slug_ml)
                    horizonte = int(p.get("dia", 0) or 0)
                    data_prevista = (datetime.fromisoformat(data_ref[:10]) + timedelta(days=horizonte)).date().isoformat()
                    self._garantir_tempo(cursor, data_prevista)

                    cursor.execute("""
                        INSERT INTO fato_previsoes_ml (
                            execucao_id, sistema_id, data_previsao, horizonte_dias, p10, p50, p90,
                            prob_sub_30, prob_sub_50, prob_acima_80
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exec_id, sis_id, data_prevista,
                        horizonte,
                        p.get("volume_cantareira_p10"),
                        p.get("volume_cantareira_p50", p.get("volume_cantareira_pct")),
                        p.get("volume_cantareira_p90"),
                        p.get("prob_cantareira_sub_30_pct"),
                        p.get("prob_cantareira_sub_50_pct"),
                        p.get("prob_cantareira_acima_80_pct")
                    ))

                # 4. Fato Previsões Meteo
                if weather and "previsoes" in weather:
                    for prev in weather["previsoes"]:
                        data_prev = prev.get("data_hora", "")[:10]
                        if data_prev:
                            self._garantir_tempo(cursor, data_prev)
                        cursor.execute("""
                            INSERT INTO fato_previsoes_meteo (
                                execucao_id, data_previsao, chuva_mm, temp_c, umidade_pct, descricao
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            exec_id,
                            data_prev,
                            prev.get("precipitacao_mm"),
                            prev.get("temperatura_c"),
                            prev.get("umidade_pct"),
                            prev.get("descricao")
                        ))

                conn.commit()
                return exec_id
        except Exception as e:
            logger.error("[Persistence] BI Error: %s", e)
            return -1
