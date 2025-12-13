"""
Exemplo de uso do sistema de pooling de tensores.
"""

import sys
import os
import torch

# Adiciona o diretório raiz ao path para permitir importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memory_optimization_systems.tensor_pooling_system.memory_pool import MemoryPool, DataType, allocate_tensor, deallocate_tensor


def example_usage():
    """Exemplo de uso do sistema de pooling de tensores."""
    print("=== Exemplo de uso do Memory Pool ===")
    
    # Criar um pool de memória com limite de 50MB
    pool = MemoryPool(max_pool_size_mb=50)
    
    print(f"Tamanho máximo do pool: {pool.max_pool_size_bytes / (1024*1024):.2f} MB")
    
    # Alocar alguns tensores
    tensor1 = pool.get_tensor((100, 100), DataType.FLOAT32)
    print(f"Alocado tensor de forma {tensor1.shape} com dtype {tensor1.dtype}")
    
    tensor2 = pool.get_tensor((50, 50, 3), DataType.FLOAT16)
    print(f"Alocado tensor de forma {tensor2.shape} com dtype {tensor2.dtype}")
    
    # Liberar os tensores de volta ao pool
    pool.release_tensor(tensor1, (100, 100))
    print("Tensor1 liberado de volta ao pool")
    
    pool.release_tensor(tensor2, (50, 50, 3))
    print("Tensor2 liberado de volta ao pool")
    
    # Obter novamente os mesmos tensores (devem vir do pool)
    tensor3 = pool.get_tensor((100, 100), DataType.FLOAT32)
    print(f"Alocado tensor3 de forma {tensor3.shape} com dtype {tensor3.dtype} (provavelmente do pool)")
    
    # Mostrar estatísticas
    stats = pool.get_stats()
    print(f"\nEstatísticas do pool:")
    print(f"- Alocações do pool: {stats['allocations_from_pool']}")
    print(f"- Alocações diretas: {stats['direct_allocations']}")
    print(f"- Liberações para o pool: {stats['deallocations_to_pool']}")
    print(f"- Taxa de acerto do pool: {stats['pool_hit_rate']:.2%}")
    print(f"- Tamanho atual do pool: {stats['current_pool_size_bytes'] / 1024:.2f} KB")
    
    # Uso do pool global
    print("\n=== Usando o pool global ===")
    global_tensor = allocate_tensor((200, 200), DataType.FLOAT32)
    print(f"Alocado tensor global de forma {global_tensor.shape}")
    
    deallocate_tensor(global_tensor, (200, 200))
    print("Tensor global liberado de volta ao pool")


if __name__ == "__main__":
    example_usage()