#!/usr/bin/env python3
"""
Script de verificação para confirmar que os componentes de otimização 
estão corretamente organizados nos diretórios apropriados em cada modelo.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple


def get_expected_optimization_dirs() -> List[str]:
    """Retorna a lista de diretórios de otimização esperados."""
    return [
        "attention",
        "fused_layers", 
        "kv_cache",
        "linear_optimizations",
        "rotary_embeddings",
        "specific_optimizations",
        "cuda_kernels",
        "tensor_parallel",
        "prefix_caching"
    ]


def check_model_optimization_structure(model_path: Path, expected_dirs: List[str]) -> Tuple[Dict[str, bool], List[str]]:
    """Verifica a estrutura de otimização de um modelo específico."""
    results = {}
    missing_dirs = []
    
    for opt_dir in expected_dirs:
        dir_path = model_path / opt_dir
        exists = dir_path.exists() and dir_path.is_dir()
        results[opt_dir] = exists
        
        if not exists:
            missing_dirs.append(opt_dir)
    
    return results, missing_dirs


def validate_model_files(model_path: Path, expected_dirs: List[str]) -> Dict[str, List[str]]:
    """Valida se há arquivos Python nos diretórios de otimização."""
    validation_results = {}
    
    for opt_dir in expected_dirs:
        dir_path = model_path / opt_dir
        if dir_path.exists():
            py_files = list(dir_path.glob("*.py"))
            validation_results[opt_dir] = [f.name for f in py_files]
        else:
            validation_results[opt_dir] = []
    
    return validation_results


def main():
    """Função principal para verificar a organização dos componentes de otimização."""
    base_paths = [
        Path("src/models"),
        Path("src/inference_pio/models")
    ]
    
    expected_dirs = get_expected_optimization_dirs()
    
    print("=" * 80)
    print("VERIFICAÇÃO DE ORGANIZAÇÃO DOS COMPONENTES DE OTIMIZAÇÃO")
    print("=" * 80)
    print(f"Diretórios de otimização esperados: {expected_dirs}")
    print()
    
    all_models_ok = True
    
    for base_path in base_paths:
        if not base_path.exists():
            print(f"Diretório base {base_path} não encontrado, pulando...")
            continue
            
        print(f"Verificando modelos em: {base_path}")
        print("-" * 60)
        
        # Encontrar modelos específicos mencionados
        target_models = ["qwen3_0_6b", "qwen3_coder_next"]
        
        for model_name in target_models:
            # Procurar o modelo em subdiretórios
            model_dirs = list(base_path.rglob(f"*{model_name}*"))
            
            for model_dir in model_dirs:
                if model_dir.is_dir():
                    print(f"\nVerificando modelo: {model_dir}")
                    
                    # Verificar estrutura de diretórios
                    structure_results, missing_dirs = check_model_optimization_structure(model_dir, expected_dirs)
                    
                    print("  Estrutura de diretórios:")
                    for opt_dir, exists in structure_results.items():
                        status = "[OK]" if exists else "[FALTA]"
                        print(f"    {status} {opt_dir}")
                    
                    # Validar arquivos nos diretórios
                    file_validation = validate_model_files(model_dir, expected_dirs)
                    
                    print("  Arquivos encontrados nos diretórios:")
                    total_files = 0
                    for opt_dir, files in file_validation.items():
                        if files:
                            print(f"    {opt_dir}: {len(files)} arquivos - {files}")
                            total_files += len(files)
                        else:
                            # Mostrar apenas diretórios vazios se não houver arquivos
                            dir_path = model_dir / opt_dir
                            if dir_path.exists():
                                print(f"    {opt_dir}: 0 arquivos (diretório existe)")
                    
                    print(f"    Total de arquivos de otimização: {total_files}")
                    
                    # Avaliar resultado geral
                    if missing_dirs:
                        print(f"  [ERRO] Diretórios faltando: {missing_dirs}")
                        all_models_ok = False
                    else:
                        print(f"  [SUCESSO] Todos os diretórios de otimização estão presentes")
                    
                    print()
    
    print("=" * 80)
    if all_models_ok:
        print("[RESULTADO FINAL] Todos os modelos têm os componentes de otimização devidamente organizados!")
    else:
        print("[RESULTADO FINAL] Alguns modelos ainda têm problemas na organização dos componentes de otimização.")
    print("=" * 80)


if __name__ == "__main__":
    main()