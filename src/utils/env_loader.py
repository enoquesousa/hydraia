"""
Carregador de Variáveis de Ambiente (.env)
==========================================
Utilitário para carregar e validar variáveis de ambiente do arquivo .env
com fallback seguro e mensagens de erro claras.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict


from src.utils.paths import HydraPaths

class EnvLoader:
    """Carrega e valida variáveis de ambiente do arquivo .env"""

    def __init__(self, env_path: Optional[str] = None):
        """
        Args:
            env_path: Caminho customizado para o arquivo .env (opcional)
        """
        if env_path is None:
            # Buscar .env SEMPRE na raiz do projeto (HydraPaths.BASE)
            self.env_path = HydraPaths.BASE / '.env'
        else:
            self.env_path = Path(env_path)

        self._carregar_env()

    def _carregar_env(self):
        """Carrega variáveis do arquivo .env para os.environ"""
        if not self.env_path.exists():
            print(f"⚠️  Arquivo .env não encontrado em: {self.env_path}")
            print(f"   Copie o arquivo .env.example para .env e configure suas chaves de API")
            return

        try:
            with open(self.env_path, 'r', encoding='utf-8') as f:
                for linha in f:
                    linha = linha.strip()

                    # Ignorar comentários e linhas vazias
                    if not linha or linha.startswith('#'):
                        continue

                    # Parse KEY=VALUE
                    if '=' in linha:
                        chave, valor = linha.split('=', 1)
                        chave = chave.strip()
                        valor = valor.strip()

                        # Só sobrescrever se não existir no ambiente
                        if chave not in os.environ:
                            os.environ[chave] = valor

        except Exception as e:
            print(f"❌ Erro ao carregar .env: {e}")

    @staticmethod
    def get(chave: str, default: Optional[str] = None, obrigatorio: bool = False) -> Optional[str]:
        """
        Obtém uma variável de ambiente.

        Args:
            chave: Nome da variável
            default: Valor padrão se não encontrado
            obrigatorio: Se True, lança exceção se não encontrado

        Returns:
            Valor da variável ou default

        Raises:
            ValueError: Se obrigatorio=True e variável não encontrada
        """
        valor = os.environ.get(chave, default)

        if obrigatorio and not valor:
            raise ValueError(
                f"Variável de ambiente '{chave}' é obrigatória mas não foi encontrada.\n"
                f"Configure-a no arquivo .env"
            )

        return valor

    @staticmethod
    def get_int(chave: str, default: int = 0) -> int:
        """Obtém variável de ambiente como inteiro"""
        valor = os.environ.get(chave)
        if not valor:
            return default

        try:
            return int(valor)
        except ValueError:
            print(f"⚠️  Erro ao converter '{chave}={valor}' para int, usando default={default}")
            return default

    @staticmethod
    def get_float(chave: str, default: float = 0.0) -> float:
        """Obtém variável de ambiente como float"""
        valor = os.environ.get(chave)
        if not valor:
            return default

        try:
            return float(valor)
        except ValueError:
            print(f"⚠️  Erro ao converter '{chave}={valor}' para float, usando default={default}")
            return default

    @staticmethod
    def get_bool(chave: str, default: bool = False) -> bool:
        """Obtém variável de ambiente como booleano"""
        valor = os.environ.get(chave, '').lower()

        if valor in ('true', '1', 'yes', 'sim'):
            return True
        elif valor in ('false', '0', 'no', 'não'):
            return False

        return default

    @staticmethod
    def validar_chaves_obrigatorias(chaves: list) -> Dict[str, bool]:
        """
        Valida se todas as chaves obrigatórias estão configuradas.

        Args:
            chaves: Lista de nomes de variáveis obrigatórias

        Returns:
            Dict com status de cada chave {chave: bool}
        """
        resultado = {}

        for chave in chaves:
            valor = os.environ.get(chave)
            configurado = bool(valor and valor.strip() and valor != f"sua_chave_{chave.lower()}_aqui")
            resultado[chave] = configurado

        return resultado

    @staticmethod
    def exibir_status_config() -> bool:
        """Exibe o status de configuração das chaves de API e retorna True se ao menos uma estiver presente."""
        chaves = [
            "OPENWEATHER_API_KEY",
            "GEMINI_ULTRA_API_KEY",
            "OPENAI_GPT5_API_KEY",
        ]
        status = EnvLoader.validar_chaves_obrigatorias(chaves)
        configuradas = 0
        for chave, ok in status.items():
            icone = "OK" if ok else "FALTANDO"
            print(f"  [{icone}] {chave}")
            if ok:
                configuradas += 1
        print(f"  {configuradas}/{len(chaves)} chaves configuradas.")
        return configuradas > 0

    @staticmethod
    def validar_configuracao_critica():
        """
        Realiza a validação rígida das variáveis críticas de ambiente.
        Interrompe a execução (SystemExit) se variáveis obrigatórias estiverem ausentes.
        """
        chaves_criticas = [
            "OPENWEATHER_API_KEY",
            "GEMINI_ULTRA_API_KEY",
            "OPENAI_GPT5_API_KEY"
        ]

        status = EnvLoader.validar_chaves_obrigatorias(chaves_criticas)
        ausentes = [chave for chave, configurado in status.items() if not configurado]

        if ausentes:
            print("\n" + "!" * 50)
            print("❌ ERRO CRÍTICO DE CONFIGURAÇÃO")
            print("!" * 50)
            print(f"As seguintes variáveis obrigatórias NÃO estão configuradas no .env:")
            for chave in ausentes:
                print(f"  - {chave}")
            print("\nO sistema não pode ser iniciado sem estas chaves.")
            print("Configure-as no arquivo .env e tente novamente.")
            print("!" * 50 + "\n")
            sys.exit(1)

        print("\n✅ Configuração de ambiente validada com sucesso.")


# Carregar .env automaticamente ao importar o módulo
_loader = EnvLoader()


# Função de conveniência para uso direto
def get_env(chave: str, default: Optional[str] = None, obrigatorio: bool = False) -> Optional[str]:
    """Atalho para EnvLoader.get()"""
    return EnvLoader.get(chave, default, obrigatorio)


if __name__ == "__main__":
    # Teste do carregador
    print("🧪 Testando EnvLoader...\n")

    loader = EnvLoader()
    loader.exibir_status_config()

    # Testar leitura de configurações opcionais
    print("\n📌 CONFIGURAÇÕES OPCIONAIS:")
    print(f"HTTP_TIMEOUT: {EnvLoader.get_int('HTTP_TIMEOUT', 30)}s")
    print(f"HTTP_RETRY_ATTEMPTS: {EnvLoader.get_int('HTTP_RETRY_ATTEMPTS', 3)}")
    print(f"LOG_LEVEL: {EnvLoader.get('LOG_LEVEL', 'INFO')}")
