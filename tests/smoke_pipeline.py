"""Smoke test end-to-end do pipeline HydraIA.

Componentes REAIS exercitados:
  - SabespCollector → API oficial v4 (live)
  - HydroLSTM       → modelo treinado em GPU (RTX 3060)
  - Qwen 2.5 14B    → síntese local via LM Studio
  - FileCache       → roundtrip + atomic write
  - Cross-validation → lógica determinística

Componentes MOCKADOS:
  - GeminiSpecialist e GPTSpecialist (chaves serão trocadas separadamente)

Uso:
    .venv/Scripts/python.exe tests/smoke_pipeline.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Forçar UTF-8 no console Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Permitir rodar como script direto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke")


def secao(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


# ─── Etapa 1 ──────────────────────────────────────────────────────────────────

def etapa_sabesp() -> dict:
    secao("1. Coleta SABESP REAL (API v4 oficial)")
    from src.api.sabesp import SabespCollector, SISTEMAS_SABESP

    t0 = time.perf_counter()
    coletor = SabespCollector(auto_save=False)
    dados = coletor.coletar_macrometropole_completa()
    elapsed = time.perf_counter() - t0

    print(f"\nColeta concluida em {elapsed:.1f}s")
    print(f"Data referencia: {dados['data_referencia']}")
    print(f"Sistemas coletados: {len(dados['sistemas'])}")
    for chave, info in dados["sistemas"].items():
        nome = SISTEMAS_SABESP.get(chave, chave)
        vol = info.get("volume_percentual")
        chuva = info.get("chuva_mm")
        print(f"  {nome:<28} {vol:>6.2f}%   chuva: {chuva} mm")

    resumo = dados.get("resumo", {})
    print(f"\nVolume medio: {resumo.get('volume_medio_percentual')}%")
    print(f"Sistemas <50%: {resumo.get('sistemas_abaixo_50pct')}")
    print(f"Sistemas <30% (criticos): {resumo.get('sistemas_criticos')}")
    return dados


# ─── Etapa 2 ──────────────────────────────────────────────────────────────────

def etapa_lstm() -> list:
    secao("2. Inferencia HydroLSTM REAL (GPU)")
    from src.data.inferencia_ts import prever
    import torch

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    t0 = time.perf_counter()
    resultado = prever(horizonte_dias=7)
    elapsed = time.perf_counter() - t0

    print(f"\nInferencia concluida em {elapsed:.2f}s")
    print(f"Sistemas previstos: {list(resultado.keys())}\n")
    print("Previsao 7 dias (v_util %):")
    print(f"  {'Dia':<6} " + " ".join(f"{s:<14}" for s in resultado.keys()))
    sistemas = list(resultado.keys())
    for i in range(7):
        linha = f"  +{i+1:<5}"
        for s in sistemas:
            v = resultado[s][i]["v_util_pct"]
            linha += f" {v:>13.2f}%"
        print(linha)

    # Converter para o formato consumido pelo orquestrador
    cantareira = resultado.get("Cantareira", [])
    previsoes_orq = []
    for i, dia in enumerate(cantareira[:7]):
        row = {"dia": float(i + 1), "chuva_mm": 0.0,
               "volume_cantareira_pct": dia["v_util_pct"]}
        for s, dias in resultado.items():
            if s != "Cantareira" and i < len(dias):
                chave = f"volume_{s.lower().replace(' ', '_')}_pct"
                row[chave] = dias[i]["v_util_pct"]
        previsoes_orq.append(row)
    return previsoes_orq


# ─── Etapa 3 ──────────────────────────────────────────────────────────────────

def etapa_orquestrador(dados_sabesp: dict, previsoes: list) -> str:
    secao("3. Orquestrador: Mocks Gemini/GPT + Cross-validation + Qwen REAL")
    from datetime import date

    from src.core.llm_specialists import LLMSpecialist
    from src.core.orquestrador import OrquestradorLLM
    from src.models.hydraia_llm import ask_local_llm

    # ── Mocks dos especialistas externos (chaves serao trocadas pelo usuario) ──
    class FakeHidro(LLMSpecialist):
        def __init__(self):
            super().__init__(api_key="mock", model="gemini-2.5-pro-mock", nome="Gemini-MOCK")
        def _invocar(self, prompt: str) -> str:
            return ('{"nivel_risco": 3, "tendencia": "estavel", '
                    '"sistemas_criticos": ["Cotia"], '
                    '"analise": "Reservatorios em faixa de atencao mas estaveis. '
                    'Cotia abaixo de 30% requer monitoramento.", '
                    '"confianca_pct": 78}')

    class FakeMeteo(LLMSpecialist):
        def __init__(self):
            super().__init__(api_key="mock", model="gpt-4o-mock", nome="GPT-MOCK")
        def _invocar(self, prompt: str) -> str:
            return ('{"nivel_risco": 2, "chuva_total_esperada_mm": 32.5, '
                    '"impacto_nos_reservatorios": "positivo", '
                    '"dias_com_chuva_significativa": 3, '
                    '"analise": "Previsao de chuvas moderadas em 3 dos 7 dias '
                    'deve mitigar risco de queda nos reservatorios.", '
                    '"confianca_pct": 82}')

    orq = OrquestradorLLM(hidro=FakeHidro(), meteo=FakeMeteo(), sintetizador=ask_local_llm)

    cantareira = dados_sabesp["sistemas"]["cantareira"]
    vol_pct = cantareira.get("volume_percentual", 0)
    if vol_pct >= 60:
        situacao = "Faixa Normal"
    elif vol_pct >= 40:
        situacao = "Faixa de Atencao"
    elif vol_pct >= 20:
        situacao = "Faixa de Alerta"
    else:
        situacao = "Faixa de Restricao (Critica)"

    estado = {
        "data_referencia": cantareira.get("data", date.today().isoformat()),
        "volume_cantareira_pct": vol_pct,
        "volume_cantareira_hm3": cantareira.get("volume_hm3"),
        "chuva_mm": cantareira.get("chuva_mm"),
        "vazao_natural_m3s": cantareira.get("vazao_natural_m3s"),
        "vazao_jusante_m3s": cantareira.get("vazao_jusante_m3s"),
        "situacao_operacional": situacao,
        "fonte_dados": "API SABESP (live)",
        "macrometropole": {
            "total_sistemas": dados_sabesp["resumo"].get("total_sistemas"),
            "volume_medio_pct": dados_sabesp["resumo"].get("volume_medio_percentual"),
            "sistemas_abaixo_50pct": dados_sabesp["resumo"].get("sistemas_abaixo_50pct"),
            "sistemas_criticos": dados_sabesp["resumo"].get("sistemas_criticos"),
        },
    }

    # Carregar contexto cientifico do TCC
    from src.utils.paths import HydraPaths
    contexto = (HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md").read_text(encoding="utf-8")

    t0 = time.perf_counter()
    relatorio = orq.gerar_relatorio(
        estado_atual=estado,
        previsoes_ml=previsoes,
        weather=None,                 # OpenWeather depende de chave; pulamos
        contexto_cientifico=contexto,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nSintese Qwen 2.5 14B concluida em {elapsed:.1f}s")
    print(f"Tamanho do relatorio: {len(relatorio)} caracteres")
    return relatorio


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    secao("HydraIA — Smoke Test End-to-End")
    print("Componentes REAIS: SABESP API + HydroLSTM (GPU) + Qwen 2.5 14B")
    print("Componentes MOCK:  Gemini, GPT (chaves a trocar pelo usuario)")

    try:
        dados = etapa_sabesp()
        previsoes = etapa_lstm()
        relatorio = etapa_orquestrador(dados, previsoes)

        secao("4. RELATORIO FINAL (gerado pelo Qwen 2.5 14B)")
        print(relatorio)
        secao("OK — Pipeline ponta-a-ponta funcionou")
        return 0

    except Exception as e:
        secao("ERRO no pipeline")
        logger.exception("Falha:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
