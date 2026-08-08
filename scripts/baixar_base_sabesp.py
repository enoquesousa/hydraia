"""
Downloader completo da base SABESP
===================================
Baixa todos os elementos × parâmetros disponíveis.
API: https://mananciais.sabesp.com.br/api/v4/elementos/{id_el}/dados/{id_param}/diario/{ini}/{fim}

Estrutura de saída:
  dados_sabesp/
    meta/
      elementos.json   ← catálogo completo
      parametros.json  ← catálogo completo
    series/
      tipo_{N}/
        el_{id}_{nome}/
          param_{id}_{nome}.json

Uso:
    python baixar_base_sabesp.py [--tipos 1,2,3] [--sem-municipios]

Por padrão baixa tipos 1–7 (sistemas, reservatórios, EEABs, túneis, ETAs, postos,
pontos hidrológicos). Tipo 11 (municípios, 653 itens) pode ser incluído com --municipios.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp

from src.utils.paths import HydraPaths

# ===========================================================================
# Configuração
# ===========================================================================

BASE_URL = "https://mananciais.sabesp.com.br/api/v4"
DATA_INICIO = "2000-01-01"
DATA_FIM = "2026-04-10"
DATA_DIR = HydraPaths.SABESP

MAX_CONCURRENT = 8          # requisições simultâneas
DELAY_ENTRE_LOTES = 0.3     # segundos entre lotes (cortesia ao servidor)
LOTE_SIZE = 40

TIPOS_PADRAO = {1, 2, 3, 4, 5, 6, 7}  # exclui tipo 11 (municípios) por padrão

TIPO_NOMES = {
    1: "sistemas",
    2: "reservatorios",
    3: "eeabs",
    4: "tuneis",
    5: "etas",
    6: "postos",
    7: "pontos_hidrologicos",
    11: "municipios",
}

# ===========================================================================
# Utilitários
# ===========================================================================

def slugify(nome: str) -> str:
    nome = nome.lower()
    nome = re.sub(r"[áàãâä]", "a", nome)
    nome = re.sub(r"[éèêë]", "e", nome)
    nome = re.sub(r"[íìîï]", "i", nome)
    nome = re.sub(r"[óòõôö]", "o", nome)
    nome = re.sub(r"[úùûü]", "u", nome)
    nome = re.sub(r"[ç]", "c", nome)
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    return nome.strip("_")


def caminho_serie(elemento: dict, parametro: dict) -> Path:
    tipo = elemento["idTipoElemento"]
    tipo_nome = TIPO_NOMES.get(tipo, f"tipo_{tipo}")
    el_slug = slugify(elemento["name"])
    pa_slug = slugify(parametro["shortName"])
    return (
        DATA_DIR
        / "series"
        / tipo_nome
        / f"el_{elemento['id']}_{el_slug}"
        / f"param_{parametro['id']}_{pa_slug}.json"
    )


def carregar_meta() -> tuple[list, list]:
    with open(DATA_DIR / "meta" / "elementos.json", encoding="utf-8") as f:
        elementos = json.load(f)["data"]
    with open(DATA_DIR / "meta" / "parametros.json", encoding="utf-8") as f:
        parametros = json.load(f)["data"]
    return elementos, parametros


# ===========================================================================
# Download
# ===========================================================================

async def baixar_serie(
    session: aiohttp.ClientSession,
    semaforo: asyncio.Semaphore,
    elemento: dict,
    parametro: dict,
) -> tuple[int, int, bool]:
    """Retorna (id_el, id_param, sucesso)."""
    id_el = elemento["id"]
    id_pa = parametro["id"]
    arquivo = caminho_serie(elemento, parametro)

    if arquivo.exists():
        return id_el, id_pa, True  # já baixado

    url = f"{BASE_URL}/elementos/{id_el}/dados/{id_pa}/diario/{DATA_INICIO}/{DATA_FIM}"

    async with semaforo:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    dados = await resp.json(content_type=None)
                    registros = dados.get("data", [])

                    # Salva mesmo séries vazias para não repetir a requisição
                    arquivo.parent.mkdir(parents=True, exist_ok=True)
                    saida = {
                        "elemento": {
                            "id": id_el,
                            "nome": elemento["name"],
                            "tipo": elemento["idTipoElemento"],
                        },
                        "parametro": {
                            "id": id_pa,
                            "nome": parametro["name"],
                            "simbolo": parametro.get("symbol", ""),
                        },
                        "periodo": {"inicio": DATA_INICIO, "fim": DATA_FIM},
                        "total_registros": len(registros),
                        "dados": registros,
                    }
                    with open(arquivo, "w", encoding="utf-8") as f:
                        json.dump(saida, f, ensure_ascii=False)
                    return id_el, id_pa, True

                else:
                    return id_el, id_pa, False

        except Exception as e:
            print(f"\n  ERRO el={id_el} param={id_pa}: {e}")
            return id_el, id_pa, False


# ===========================================================================
# Main
# ===========================================================================

async def main(tipos_incluir: set[int]):
    elementos, parametros = carregar_meta()

    # Filtrar tipos
    elementos_sel = [e for e in elementos if e["idTipoElemento"] in tipos_incluir]

    # Construir todas as combinações (elemento, parametro)
    combinacoes = []
    for el in elementos_sel:
        for pa in parametros:
            combinacoes.append((el, pa))

    total = len(combinacoes)
    ja_existem = sum(1 for el, pa in combinacoes if caminho_serie(el, pa).exists())
    faltando = [(el, pa) for el, pa in combinacoes if not caminho_serie(el, pa).exists()]

    print(f"Tipos selecionados: {sorted(tipos_incluir)}")
    print(f"Elementos: {len(elementos_sel)}  ×  Parâmetros: {len(parametros)}")
    print(f"Total de séries: {total}")
    print(f"Já baixadas: {ja_existem}  |  Faltando: {len(faltando)}")

    if not faltando:
        print("Nada a baixar.")
        return

    semaforo = asyncio.Semaphore(MAX_CONCURRENT)
    erros = []
    baixados = 0

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 2)
    headers = {"User-Agent": "hydraia-research/1.0"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for i in range(0, len(faltando), LOTE_SIZE):
            lote = faltando[i : i + LOTE_SIZE]
            tarefas = [baixar_serie(session, semaforo, el, pa) for el, pa in lote]
            resultados = await asyncio.gather(*tarefas)

            for id_el, id_pa, ok in resultados:
                if ok:
                    baixados += 1
                else:
                    erros.append((id_el, id_pa))

            progresso = ja_existem + baixados
            pct = progresso / total * 100
            lote_num = i // LOTE_SIZE + 1
            total_lotes = (len(faltando) + LOTE_SIZE - 1) // LOTE_SIZE
            print(
                f"  [{progresso}/{total}] {pct:.1f}%  lote {lote_num}/{total_lotes}"
                f"  erros: {len(erros)}        ",
                end="\r",
            )

            if DELAY_ENTRE_LOTES > 0:
                await asyncio.sleep(DELAY_ENTRE_LOTES)

    print(f"\nConcluído. Baixadas: {baixados}  Erros: {len(erros)}")
    if erros:
        print(f"Séries com erro ({len(erros)}) — rode novamente para retentar:")
        for id_el, id_pa in erros[:15]:
            print(f"  elemento={id_el}  parâmetro={id_pa}")
        if len(erros) > 15:
            print(f"  ... e mais {len(erros) - 15}")


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader base SABESP")
    parser.add_argument(
        "--tipos",
        default="1,2,3,4,5,6,7",
        help="Tipos de elemento a incluir (default: 1,2,3,4,5,6,7). Use --tipos 1,2,11 para incluir municípios.",
    )
    parser.add_argument(
        "--municipios",
        action="store_true",
        help="Incluir tipo 11 (653 municípios). Aumenta muito o volume.",
    )
    args = parser.parse_args()

    tipos = {int(t) for t in args.tipos.split(",") if t.strip()}
    if args.municipios:
        tipos.add(11)

    t0 = time.time()
    asyncio.run(main(tipos))
    elapsed = time.time() - t0
    print(f"Tempo total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
