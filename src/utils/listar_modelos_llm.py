"""Lista modelos disponíveis nas contas OpenAI e Google Gemini.

Requisitos:
- OPENAI_GPT5_API_KEY no .env (usado pela classe EnvLoader)
- GEMINI_ULTRA_API_KEY no .env

Uso:
    python listar_modelos_llm.py
"""

import sys
from typing import List, Dict, Any

import requests

from src.utils.env_loader import EnvLoader
from src.utils.http_security import exigir_resposta_sem_redirecionamento


def listar_modelos_openai() -> List[Dict[str, Any]]:
    """Retorna a lista de modelos visíveis para a chave OpenAI.

    Usa o endpoint /v1/models, compatível com a maioria das chaves atuais.
    """

    api_key = EnvLoader.get("OPENAI_GPT5_API_KEY")
    if not api_key:
        print("[OpenAI] OPENAI_GPT5_API_KEY não encontrada no .env")
        return []

    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=False)
        exigir_resposta_sem_redirecionamento(resp)
        resp.raise_for_status()
        data = resp.json()
        modelos = data.get("data", [])
        # Ordenar por id para uma saída mais estável.
        modelos.sort(key=lambda m: m.get("id", ""))
        return modelos
    except Exception:
        print("[OpenAI] Erro ao listar modelos: falha HTTP ou resposta invalida")
        return []


def listar_modelos_gemini() -> List[Dict[str, Any]]:
    """Retorna a lista de modelos visíveis para a chave Gemini."""

    api_key = EnvLoader.get("GEMINI_ULTRA_API_KEY")
    if not api_key:
        print("[Gemini] GEMINI_ULTRA_API_KEY não encontrada no .env")
        return []

    # Endpoint oficial de listagem de modelos do Gemini
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key}

    try:
        resp = requests.get(url, params=params, timeout=15, allow_redirects=False)
        exigir_resposta_sem_redirecionamento(resp)
        resp.raise_for_status()
        data = resp.json()
        modelos = data.get("models", [])
        # Ordenar por name.
        modelos.sort(key=lambda m: m.get("name", ""))
        return modelos
    except Exception:
        print("[Gemini] Erro ao listar modelos: falha HTTP ou resposta invalida")
        return []


def main() -> None:
    print("🚀 Listando modelos disponíveis para as chaves configuradas\n")

    print("================ OpenAI =================")
    modelos_openai = listar_modelos_openai()
    if not modelos_openai:
        print("Nenhum modelo retornado ou erro na chamada.")
    else:
        for m in modelos_openai:
            mid = m.get("id", "<sem id>")
            created = m.get("created")
            owned_by = m.get("owned_by")
            print(f"- {mid} | owned_by={owned_by} | created={created}")

    print("\n================ Gemini =================")
    modelos_gemini = listar_modelos_gemini()
    if not modelos_gemini:
        print("Nenhum modelo retornado ou erro na chamada.")
    else:
        for m in modelos_gemini:
            name = m.get("name", "<sem name>")
            display = m.get("displayName", "")
            desc = m.get("description", "")
            print(f"- {name} | {display}")
            if desc:
                print(f"    {desc[:120]}{'...' if len(desc) > 120 else ''}")

    print("\n✅ Finalizado.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(1)
