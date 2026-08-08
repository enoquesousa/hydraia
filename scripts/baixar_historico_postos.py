"""
Downloader histórico de dados de postos SABESP
===============================================
API: https://mananciais.sabesp.com.br/api/v4/postos/dados/resumo-diario/{data}
Período: 2000-01-01 até hoje

Uso:
    python baixar_historico_postos.py

Salva os JSONs em: dados_sabesp/postos_historico/{YYYY}/{YYYY-MM-DD}.json
"""

import asyncio
import json
import time
from datetime import date, timedelta
from pathlib import Path

import aiohttp

from src.utils.paths import HydraPaths

BASE_URL = "https://mananciais.sabesp.com.br/api/v4/postos/dados/resumo-diario/{data}"
DATA_DIR = HydraPaths.SABESP / "postos_historico"
DATA_INICIO = date(2000, 1, 1)
DATA_FIM = date(2026, 4, 10)

# Controle de concorrência — respeita o servidor
MAX_CONCURRENT = 5
DELAY_ENTRE_LOTES = 0.5  # segundos entre lotes


def gerar_datas(inicio: date, fim: date):
    d = inicio
    while d <= fim:
        yield d
        d += timedelta(days=1)


def caminho_arquivo(d: date) -> Path:
    return DATA_DIR / str(d.year) / f"{d.isoformat()}.json"


def datas_faltando() -> list[date]:
    todas = list(gerar_datas(DATA_INICIO, DATA_FIM))
    faltando = [d for d in todas if not caminho_arquivo(d).exists()]
    return faltando


async def baixar_dia(session: aiohttp.ClientSession, d: date, semaforo: asyncio.Semaphore) -> tuple[date, bool]:
    url = BASE_URL.format(data=d.isoformat())
    async with semaforo:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    dados = await resp.json(content_type=None)
                    arquivo = caminho_arquivo(d)
                    arquivo.parent.mkdir(parents=True, exist_ok=True)
                    with open(arquivo, "w", encoding="utf-8") as f:
                        json.dump(dados, f, ensure_ascii=False)
                    return d, True
                elif resp.status == 404:
                    # Dia sem dados — salva sentinela vazio para não repetir
                    arquivo = caminho_arquivo(d)
                    arquivo.parent.mkdir(parents=True, exist_ok=True)
                    with open(arquivo, "w", encoding="utf-8") as f:
                        json.dump({"status": 404, "data": d.isoformat(), "dados": None}, f)
                    return d, True
                else:
                    return d, False
        except Exception as e:
            print(f"  ERRO {d}: {e}")
            return d, False


async def main():
    faltando = datas_faltando()
    total = (DATA_FIM - DATA_INICIO).days + 1
    ja_existem = total - len(faltando)

    print(f"Período: {DATA_INICIO} → {DATA_FIM}  ({total} dias)")
    print(f"Já baixados: {ja_existem}  |  Faltando: {len(faltando)}")

    if not faltando:
        print("Nada a baixar.")
        return

    semaforo = asyncio.Semaphore(MAX_CONCURRENT)
    erros = []
    baixados = 0

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    headers = {"User-Agent": "hydraia-research/1.0"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        # Processa em lotes para exibir progresso
        lote_size = 50
        for i in range(0, len(faltando), lote_size):
            lote = faltando[i : i + lote_size]
            tarefas = [baixar_dia(session, d, semaforo) for d in lote]
            resultados = await asyncio.gather(*tarefas)

            for d, ok in resultados:
                if ok:
                    baixados += 1
                else:
                    erros.append(d)

            progresso = ja_existem + baixados
            pct = progresso / total * 100
            print(f"  [{progresso}/{total}] {pct:.1f}%  — lote {i // lote_size + 1}  erros acumulados: {len(erros)}", end="\r")

            if DELAY_ENTRE_LOTES > 0:
                await asyncio.sleep(DELAY_ENTRE_LOTES)

    print(f"\nConcluído. Baixados: {baixados}  Erros: {len(erros)}")
    if erros:
        print("Datas com erro (podem ser reprocessadas rodando o script novamente):")
        for d in erros[:20]:
            print(f"  {d}")
        if len(erros) > 20:
            print(f"  ... e mais {len(erros) - 20}")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Tempo total: {time.time() - t0:.1f}s")
