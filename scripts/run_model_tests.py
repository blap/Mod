#!/usr/bin/env python3
"""
Script de exemplo para executar testes específicos de modelos.

Este script demonstra como executar testes para modelos específicos
usando a nova estrutura de testes organizada por modelo.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_model_tests(model_name=None, test_type=None, verbose=False):
    """
    Executa testes para um modelo específico ou todos os modelos.

    Args:
        model_name (str): Nome do modelo para testar (ex: 'qwen3_0_6b')
        test_type (str): Tipo de teste ('unit', 'integration', 'performance')
        verbose (bool): Se True, mostra saída detalhada
    """
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")

    # Define o caminho dos testes com base nos parâmetros
    if model_name:
        test_path = f"tests/models/{model_name}"

        if test_type:
            test_path = f"{test_path}/{test_type}"

        cmd.append(test_path)
    else:
        # Executar todos os testes de modelos
        cmd.append("tests/models")

    # Adicionar marcadores para filtrar testes
    if test_type:
        cmd.extend(["-m", test_type])

    print(f"Executando comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar testes: {e}")
        return False


def list_available_models():
    """Lista todos os modelos com testes disponíveis."""
    models_dir = Path("tests/models")
    if not models_dir.exists():
        print("Diretório tests/models não encontrado!")
        return []

    models = [
        d.name for d in models_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
    ]
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(
        description="Executar testes específicos de modelos"
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Nome do modelo para testar (ex: qwen3_0_6b). Se omitido, testa todos os modelos.",
    )
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance"],
        help="Tipo de teste a executar",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Mostrar saída detalhada"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="Listar modelos disponíveis"
    )

    args = parser.parse_args()

    if args.list:
        models = list_available_models()
        print("Modelos disponíveis para teste:")
        for model in models:
            print(f"  - {model}")
        return

    if args.model_name and args.model_name not in list_available_models():
        print(f"Modelo '{args.model_name}' não encontrado!")
        print("Use --list para ver os modelos disponíveis.")
        return

    success = run_model_tests(args.model_name, args.type, args.verbose)

    if success:
        print("Testes concluídos com sucesso!")
    else:
        print("Alguns testes falharam!")
        sys.exit(1)


if __name__ == "__main__":
    main()
