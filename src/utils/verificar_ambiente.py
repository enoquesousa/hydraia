"""Script de verificação completa do ambiente HydraIA.

Verifica:
- PyTorch com CUDA
- LM Studio rodando
- Arquivo .md de contexto
- Estrutura de pastas

Execute antes de rodar o pipeline principal.
"""

import sys
from pathlib import Path

import requests
import torch


def verificar_pytorch_cuda():
    """Verifica se PyTorch está instalado e detectando a GPU."""
    print("\n🔍 Verificando PyTorch + CUDA...")

    try:
        versao = torch.__version__
        cuda_disponivel = torch.cuda.is_available()

        print(f"   ✅ PyTorch {versao} instalado")

        if cuda_disponivel:
            gpu_nome = torch.cuda.get_device_name(0)
            gpu_memoria = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ CUDA disponível")
            print(f"   ✅ GPU detectada: {gpu_nome}")
            print(f"   ✅ VRAM total: {gpu_memoria:.1f} GB")
            return True
        else:
            print("   ⚠️  CUDA não disponível (rodará na CPU)")
            return False

    except Exception as e:
        print(f"   ❌ Erro ao verificar PyTorch: {e}")
        return False


def verificar_lm_studio():
    """Verifica se o LM Studio está rodando e acessível."""
    print("\n🔍 Verificando LM Studio...")

    try:
        response = requests.get(
            "http://localhost:1234/v1/models",
            timeout=5
        )

        if response.status_code == 200:
            modelos = response.json()
            if modelos.get("data"):
                print("   ✅ LM Studio rodando e acessível")
                print(f"   ✅ Modelos carregados: {len(modelos['data'])}")

                for modelo in modelos["data"]:
                    print(f"      - {modelo.get('id', 'desconhecido')}")
                return True
            else:
                print("   ⚠️  LM Studio rodando mas nenhum modelo carregado")
                print("      Carregue o Qwen3.5-35B-A3B no LM Studio")
                return False
        else:
            print(f"   ❌ LM Studio respondeu com código {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("   ❌ LM Studio não está rodando")
        print("\n   Como resolver:")
        print("   1. Abra o LM Studio")
        print("   2. Carregue o modelo Qwen3.5-35B-A3B")
        print("   3. Vá em 'Developer' → 'Local Server'")
        print("   4. Clique em 'Start Server' (porta 1234)")
        return False

    except Exception as e:
        print(f"   ❌ Erro ao verificar LM Studio: {e}")
        return False


from src.utils.paths import HydraPaths

def verificar_arquivos_contexto():
    """Verifica se o arquivo .md de contexto existe."""
    print("\n🔍 Verificando arquivos de contexto...")

    arquivo_md = HydraPaths.BASE / "docs" / "Crises_Hidricas_SP.md"

    if arquivo_md.exists():
        tamanho_kb = arquivo_md.stat().st_size / 1024
        print(f"   ✅ Arquivo de contexto encontrado")
        print(f"      {arquivo_md}")
        print(f"      Tamanho: {tamanho_kb:.1f} KB")
        return True
    else:
        print(f"   ❌ Arquivo de contexto não encontrado:")
        print(f"      {arquivo_md}")
        print("\n   Como resolver:")
        print("   1. Certifique-se de que o arquivo .md está na pasta docs/")
        return False


def verificar_estrutura_pastas():
    """Verifica se as pastas necessárias de armazenamento existem."""
    print("\n🔍 Verificando estrutura de armazenamento (storage/)...")

    pastas_necessarias = [
        HydraPaths.SABESP,
        HydraPaths.DATASETS,
        HydraPaths.HISTORY,
        HydraPaths.LLM,
        HydraPaths.TIME_SERIES,
        HydraPaths.REPORTS
    ]

    todas_ok = True
    for pasta in pastas_necessarias:
        if pasta.exists():
            print(f"   ✅ {pasta.relative_to(HydraPaths.BASE)}/ encontrada")
        else:
            print(f"   ⚠️  {pasta.relative_to(HydraPaths.BASE)}/ não encontrada (criando...)")
            pasta.mkdir(parents=True, exist_ok=True)
            print(f"      ✅ Pasta criada")

    return todas_ok


def main():
    """Executa todas as verificações."""
    print("=" * 60)
    print("  VERIFICAÇÃO DO AMBIENTE HYDRAIA")
    print("=" * 60)

    # Garantir que os diretórios base existam
    HydraPaths.ensure_dirs()

    resultados = {
        "pytorch_cuda": verificar_pytorch_cuda(),
        "lm_studio": verificar_lm_studio(),
        "arquivos": verificar_arquivos_contexto(),
        "estrutura": verificar_estrutura_pastas(),
    }

    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)

    if resultados["pytorch_cuda"]:
        print("✅ PyTorch + GPU: OK")
    else:
        print("⚠️  PyTorch + GPU: Problema detectado")

    if resultados["lm_studio"]:
        print("✅ LM Studio: OK")
    else:
        print("❌ LM Studio: NÃO DISPONÍVEL")

    if resultados["arquivos"]:
        print("✅ Arquivos de contexto: OK")
    else:
        print("❌ Arquivos de contexto: FALTANDO")

    if resultados["estrutura"]:
        print("✅ Estrutura de pastas: OK")
    else:
        print("⚠️  Estrutura de pastas: Ajustada")

    print("\n" + "=" * 60)

    # Conclusão
    tudo_ok = all(resultados.values())

    if tudo_ok:
        print("✅ AMBIENTE PRONTO!")
        print("\nVocê pode executar o pipeline agora:")
        print("   python integracao.py")
        return 0
    else:
        print("⚠️  AMBIENTE COM PENDÊNCIAS")
        print("\nResolva os problemas acima antes de executar o pipeline.")

        if not resultados["lm_studio"]:
            print("\n💡 DICA: O problema mais comum é o LM Studio não estar rodando.")
            print("   Abra o LM Studio, carregue o Qwen3.5-35B-A3B e ative o servidor local.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
