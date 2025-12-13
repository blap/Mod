"""
Script de teste de desempenho para o MemoryDefragmenter
Específico para hardware: Intel i5-10210U + NVIDIA SM61 + NVMe SSD
"""

import torch
import time
import psutil
from memory_optimization_systems.memory_defragmentation_system.memory_defragmenter import (
    MemoryDefragmenter, 
    create_optimized_defragmenter
)
import matplotlib.pyplot as plt
import numpy as np


def test_performance_on_target_hardware():
    """
    Testa o desempenho do MemoryDefragmenter no hardware alvo.
    """
    print("Iniciando testes de desempenho para hardware alvo: Intel i5-10210U + NVIDIA SM61")
    
    # Verifica disponibilidade de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo detectado: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        print(f"GPU: {gpu_name}")
        print(f"Memória total da GPU: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # Cria defragmenter otimizado para o hardware
    defrag = create_optimized_defragmenter("intel_i5_nvidia_sm61")
    
    # Teste 1: Criação e liberação de tensores para causar fragmentação
    print("\n=== Teste 1: Criação de fragmentação ===")
    start_time = time.time()
    
    # Cria muitos pequenos tensores para causar fragmentação
    tensors = []
    for i in range(50):
        # Cria tensores de tamanhos variados
        size = np.random.randint(100, 1000)  # Tamanhos entre 100x100 e 1000x1000
        tensor = torch.randn(size, size, device=defrag.device)
        tensors.append(tensor)
    
    # Libera alguns tensores aleatoriamente para criar fragmentação
    import random
    indices_to_free = random.sample(range(len(tensors)), len(tensors) // 2)
    for idx in sorted(indices_to_free, reverse=True):
        del tensors[idx]
    
    frag_status_before = defrag.get_fragmentation_status()
    print(f"Fragmentação antes da desfragmentação: {frag_status_before.fragmentation_ratio:.2f}")
    print(f"Blocos livres antes: {frag_status_before.free_blocks_count}")
    print(f"Maior bloco livre antes: {frag_status_before.largest_free_block / 1024 / 1024:.2f} MB")
    
    # Teste 2: Desfragmentação
    print("\n=== Teste 2: Desfragmentação ===")
    defrag_start = time.time()
    defrag.defragment()
    defrag_time = time.time() - defrag_start
    
    frag_status_after = defrag.get_fragmentation_status()
    print(f"Fragmentação após desfragmentação: {frag_status_after.fragmentation_ratio:.2f}")
    print(f"Blocos livres após: {frag_status_after.free_blocks_count}")
    print(f"Maior bloco livre após: {frag_status_after.largest_free_block / 1024 / 1024:.2f} MB")
    print(f"Tempo de desfragmentação: {defrag_time:.4f}s")
    
    # Teste 3: Estresse de memória
    print("\n=== Teste 3: Estresse de memória ===")
    stress_start = time.time()
    
    # Cria um padrão de alocação/liberação intensiva
    stress_tensors = []
    for cycle in range(10):
        # Aloca vários tensores
        cycle_tensors = []
        for i in range(20):
            size = np.random.randint(500, 1500)
            tensor = torch.randn(size, size, device=defrag.device)
            cycle_tensors.append(tensor)
        
        # Libera metade aleatoriamente
        to_keep = random.sample(cycle_tensors, len(cycle_tensors) // 2)
        stress_tensors.extend(to_keep)
        
        # Libera os outros
        for tensor in cycle_tensors:
            if tensor not in to_keep:
                del tensor
    
    stress_time = time.time() - stress_start
    print(f"Tempo para operações de estresse: {stress_time:.4f}s")
    
    # Verifica status após estresse
    stress_status = defrag.get_fragmentation_status()
    print(f"Fragmentação após estresse: {stress_status.fragmentation_ratio:.2f}")
    print(f"Blocos livres após estresse: {stress_status.free_blocks_count}")
    
    # Teste 4: Monitoramento contínuo
    print("\n=== Teste 4: Monitoramento contínuo ===")
    
    # Coleta métricas ao longo do tempo
    metrics_over_time = []
    
    def metrics_callback(health):
        metrics_over_time.append({
            'timestamp': time.time(),
            'fragmentation_level': health.fragmentation_level,
            'utilization': health.utilization_percentage,
            'available_memory': health.available_memory
        })
    
    # Inicia monitoramento
    defrag.continuous_monitoring_loop(interval=0.1, callback=metrics_callback)
    
    # Executa algumas operações durante o monitoramento
    for i in range(100):
        size = np.random.randint(100, 500)
        temp_tensor = torch.randn(size, size, device=defrag.device)
        del temp_tensor
    
    # Para o monitoramento
    defrag.stop_continuous_monitoring()
    
    print(f"Métricas coletadas: {len(metrics_over_time)} amostras")
    if metrics_over_time:
        avg_utilization = np.mean([m['utilization'] for m in metrics_over_time])
        max_utilization = max([m['utilization'] for m in metrics_over_time])
        print(f"Utilização média de memória: {avg_utilization:.2f}%")
        print(f"Utilização máxima de memória: {max_utilization:.2f}%")
    
    # Teste 5: Estatísticas finais
    print("\n=== Teste 5: Estatísticas finais ===")
    stats = defrag.get_statistics()
    print(f"Chamadas de desfragmentação: {stats['defragmentation_calls']}")
    print(f"Operações de cópia realizadas: {stats['total_copy_operations']}")
    print(f"Tempo total de cópia: {stats['total_copy_time']:.4f}s")
    
    # Saúde final da memória
    final_health = defrag.get_memory_health()
    print(f"Nível de fragmentação final: {final_health.fragmentation_level}")
    print(f"Utilização final de memória: {final_health.utilization_percentage:.2f}%")
    
    # Teste 6: Comparação de desempenho com e sem otimização
    print("\n=== Teste 6: Comparação de desempenho ===")
    
    # Teste com o defragmenter otimizado
    optimized_start = time.time()
    for i in range(30):
        size = np.random.randint(200, 800)
        temp_tensor = torch.randn(size, size, device=defrag.device)
        del temp_tensor
        if i % 10 == 0:  # A cada 10 operações, tenta desfragmentar se necessário
            defrag.force_defragmentation_check()
    optimized_time = time.time() - optimized_start
    
    print(f"Tempo com defragmenter otimizado: {optimized_time:.4f}s")
    
    # Teste sem defragmenter (usando PyTorch diretamente)
    direct_start = time.time()
    for i in range(30):
        size = np.random.randint(200, 800)
        temp_tensor = torch.randn(size, size, device=device)
        del temp_tensor
    direct_time = time.time() - direct_start
    
    print(f"Tempo sem defragmenter: {direct_time:.4f}s")
    print(f"Diferença percentual: {((optimized_time - direct_time) / direct_time * 100):.2f}%")
    
    # Resultados finais
    print("\n=== Resultados Finais ===")
    print(f"Tempo total de execução: {time.time() - start_time:.4f}s")
    print(f"Hardware: Intel i5-10210U + NVIDIA SM61")
    print(f"Desempenho do defragmenter: {'ACEITÁVEL' if defrag_time < 0.1 else 'NECESSITA OTIMIZAÇÃO'}")
    print(f"Nível de fragmentação final: {final_health.fragmentation_level.upper()}")
    
    # Verifica se o desempenho está dentro dos parâmetros aceitáveis para o hardware
    if defrag_time < 0.1 and final_health.fragmentation_level in ['low', 'medium']:
        print("✓ Sistema de desfragmentação funcionando adequadamente no hardware alvo")
        return True
    else:
        print("⚠ Otimizações adicionais podem ser necessárias para o hardware alvo")
        return False


def run_comprehensive_benchmark():
    """
    Executa um benchmark abrangente do sistema de desfragmentação.
    """
    print("Executando benchmark abrangente do MemoryDefragmenter...")
    
    results = test_performance_on_target_hardware()
    
    if results:
        print("\n✓ Todos os testes de desempenho passaram com sucesso!")
        print("O sistema de desfragmentação está otimizado para o hardware alvo.")
    else:
        print("\n⚠ Alguns testes indicam que otimizações adicionais são necessárias.")
    
    return results


if __name__ == "__main__":
    run_comprehensive_benchmark()