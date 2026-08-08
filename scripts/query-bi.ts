/**
 * Exemplo de consumo do HydraIA BI (SQLite) em TypeScript.
 * Usa better-sqlite3 para leituras síncronas rápidas (ideal para SSR/Edge).
 *
 * Execução:
 *   pnpm add better-sqlite3
 *   ts-node scripts/query-bi.ts
 */
import Database from 'better-sqlite3';
import path from 'node:path';

import type { FatoMedicao, DimSistema, DimTempo, FatoPrevisaoML } from '../storage/hydraia-types';

const DB_PATH = path.resolve(__dirname, '..', 'storage', 'hydraia_bi.db');

function main() {
  const db = new Database(DB_PATH, { readonly: true });

  // Exemplo 1: Últimas medições com anomalia e estação do ano
  const stmtSnapshot = db.prepare(`
    SELECT m.*, s.nome as sistema_nome, t.estacao
    FROM fato_medicoes m
    JOIN dim_sistemas s ON m.sistema_id = s.id
    JOIN dim_tempo t ON m.data = t.data
    ORDER BY m.data DESC, s.nome ASC
    LIMIT 20
  `);
  const snapshots = stmtSnapshot.all() as (FatoMedicao & { sistema_nome: string; estacao: DimTempo['estacao'] })[];

  // Exemplo 2: Faixas probabilísticas da HydroLSTM (p10/p50/p90) para gráfico de faixa
  const stmtBands = db.prepare(`
    SELECT p.data_previsao, s.nome as sistema_nome, p.p10, p.p50, p.p90,
           p.prob_sub_30, p.prob_sub_50, p.prob_acima_80
    FROM fato_previsoes_ml p
    JOIN dim_sistemas s ON p.sistema_id = s.id
    ORDER BY p.data_previsao ASC
  `);
  const bands = stmtBands.all() as (FatoPrevisaoML & { sistema_nome: string })[];

  console.log('== Últimas medições ==');
  console.table(snapshots);

  console.log('\n== Faixas HydroLSTM ==');
  console.table(bands);

  db.close();
}

main();
