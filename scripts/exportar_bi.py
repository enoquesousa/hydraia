import sqlite3
import pandas as pd
from src.utils.persistence import DB_PATH

def exportar_csv_bi():
    print(f"Conectando ao banco de BI: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Exemplo de Query de BI: Volume por Sistema com Estação do Ano
    query = """
    SELECT
        m.data,
        s.nome as sistema,
        m.volume_pct,
        m.anomalia_zscore,
        t.estacao
    FROM fato_medicoes m
    JOIN dim_sistemas s ON m.sistema_id = s.id
    JOIN dim_tempo t ON m.data = t.data
    ORDER BY m.data DESC, s.nome ASC
    """

    df = pd.read_sql_query(query, conn)
    output_path = "storage/outputs/bi_export.csv"
    df.to_csv(output_path, index=False)
    print(f"Exportado com sucesso para: {output_path}")
    print("\nPreview dos dados (Top 5):")
    print(df.head())

    conn.close()

if __name__ == "__main__":
    exportar_csv_bi()
