#!/usr/bin/env python3
"""
Script para organizar os componentes de otimização em diretórios específicos dentro de cada modelo.
Este script verifica se os componentes de otimização estão corretamente organizados 
nos diretórios apropriados em cada modelo.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict


def get_optimization_directories() -> List[str]:
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


def find_model_directories(base_path: str) -> List[Path]:
    """Encontra diretórios de modelos em potencial."""
    model_dirs = []
    
    # Procura por padrões de nomes de modelos específicos
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if "qwen3_0_6b" in d or "qwen3_coder_next" in d:
                model_dirs.append(Path(root) / d)
                
    return model_dirs


def create_missing_directories(model_path: Path, optimization_dirs: List[str]) -> List[str]:
    """Cria diretórios de otimização que não existem."""
    created_dirs = []
    
    for opt_dir in optimization_dirs:
        dir_path = model_path / opt_dir
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
            print(f"Criado diretório: {dir_path}")
            
    return created_dirs


def verify_optimization_structure(model_path: Path, optimization_dirs: List[str]) -> Dict[str, bool]:
    """Verifica se todos os diretórios de otimização existem no modelo."""
    results = {}
    
    for opt_dir in optimization_dirs:
        dir_path = model_path / opt_dir
        exists = dir_path.exists() and dir_path.is_dir()
        results[opt_dir] = exists
        
    return results


def move_optimization_files_to_correct_dirs(model_path: Path, optimization_dirs: List[str]):
    """Move arquivos de otimização para os diretórios corretos."""
    # Encontrar arquivos de otimização espalhados
    for file_path in model_path.rglob("*.py"):
        if file_path.parent == model_path:
            # Arquivo está no diretório raiz do modelo, talvez precise ser movido
            
            # Verifica se o nome do arquivo sugere pertencer a algum componente de otimização
            filename = file_path.name.lower()
            
            target_dir = None
            for opt_dir in optimization_dirs:
                if opt_dir in filename:
                    target_dir = model_path / opt_dir
                    break
            
            # Se encontrarmos um diretório apropriado, movemos o arquivo
            if target_dir:
                target_path = target_dir / file_path.name
                print(f"Movendo {file_path} para {target_path}")
                shutil.move(str(file_path), str(target_path))


def main():
    """Função principal para organizar os componentes de otimização."""
    base_paths = [
        Path("src/models"),
        Path("src/inference_pio/models")
    ]
    
    optimization_dirs = get_optimization_directories()
    
    print("Iniciando organização dos componentes de otimização...")
    print(f"Diretórios de otimização esperados: {optimization_dirs}")
    print()
    
    for base_path in base_paths:
        if not base_path.exists():
            print(f"Diretório base {base_path} não encontrado, pulando...")
            continue
            
        print(f"Procurando modelos em: {base_path}")
        
        model_dirs = find_model_directories(str(base_path))
        
        for model_dir in model_dirs:
            print(f"\nProcessando modelo: {model_dir}")
            
            # Verifica estrutura atual
            current_structure = verify_optimization_structure(model_dir, optimization_dirs)
            
            print("  Estrutura atual:")
            missing_dirs = []
            for opt_dir, exists in current_structure.items():
                status = "[OK]" if exists else "[MISSING]"
                print(f"    {status} {opt_dir}")
                if not exists:
                    missing_dirs.append(opt_dir)
            
            # Cria diretórios faltantes
            if missing_dirs:
                print(f"  Criando {len(missing_dirs)} diretórios faltantes...")
                created = create_missing_directories(model_dir, missing_dirs)
                if created:
                    print(f"    Diretórios criados: {created}")
            
            # Move arquivos para os diretórios corretos
            print("  Movendo arquivos para diretórios apropriados...")
            move_optimization_files_to_correct_dirs(model_dir, optimization_dirs)
            
            # Verifica novamente após as alterações
            final_structure = verify_optimization_structure(model_dir, optimization_dirs)
            all_present = all(final_structure.values())
            
            if all_present:
                print(f"  [OK] Todos os diretórios de otimização estão presentes para {model_dir.name}")
            else:
                missing = [k for k, v in final_structure.items() if not v]
                print(f"  [ERROR] Ainda faltam diretórios: {missing}")
    
    print("\nOrganização concluída!")


if __name__ == "__main__":
    main()