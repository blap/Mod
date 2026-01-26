"""
Script para configurar imports corretos para execução de benchmarks de comparação
entre modelos originais e modificados.
"""
import sys
import os
from pathlib import Path

def setup_import_paths():
    """Configura os caminhos de import para permitir carregar modelos originais e modificados."""
    # Caminho absoluto para o diretório raiz do projeto
    project_root = Path(__file__).parent.parent.absolute()  # Go up one level to project root

    # Adiciona os diretórios necessários ao sys.path
    src_path = project_root / "src"

    # Garante que os caminhos estejam no início do sys.path
    paths_to_add = [str(src_path)]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Configura o PYTHONPATH para que os imports relativos funcionem corretamente
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = f"{str(src_path)}:{os.environ['PYTHONPATH']}"
    else:
        os.environ['PYTHONPATH'] = f"{str(src_path)}"

    print(f"Caminhos adicionados ao sys.path: {paths_to_add}")
    print(f"PYTHONPATH configurado: {os.environ['PYTHONPATH']}")

if __name__ == "__main__":
    setup_import_paths()
    print("Configuração de imports concluída.")