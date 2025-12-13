# Buddy Allocator System for Qwen3-VL

O Buddy Allocator System é uma implementação otimizada de um alocador de memória baseado no algoritmo Buddy, especialmente projetado para o projeto Qwen3-VL. Este sistema gerencia eficientemente alocações de memória de diferentes tamanhos com o objetivo de minimizar a fragmentação e maximizar o desempenho em hardware específico.

## Características

- **Árvore binária para rastreamento de blocos**: Implementação eficiente usando uma estrutura de árvore binária para rastrear blocos livres e ocupados
- **Alocação e desalocação com coalescing**: Combinação eficiente de blocos irmãos para reduzir fragmentação externa
- **Gerenciamento de blocos de diferentes tamanhos**: Suporte para alocação de blocos de tamanhos que são potências de 2
- **Thread safety**: Operações seguras em ambientes multithread usando locks
- **Estatísticas de desempenho e fragmentação**: Monitoramento detalhado de uso e fragmentação de memória
- **Interface amigável para tensores PyTorch**: Integração direta com tensores PyTorch para uso em modelos de linguagem
- **Otimizações de hardware**: Especificamente otimizado para Intel i5-10210U + NVIDIA SM61 + NVMe SSD

## Componentes

### BuddyAllocator
Classe principal que implementa o algoritmo Buddy para gerenciamento de memória. Inclui:

- Alocação e desalocação de blocos de memória
- Coalescing de blocos livres adjacentes
- Estatísticas de desempenho
- Thread safety

### PyTorchBuddyAllocator
Extensão do BuddyAllocator com integração PyTorch para alocação direta de tensores.

### OptimizedBuddyAllocator
Versão otimizada com parâmetros específicos para o hardware alvo (Intel i5-10210U + NVIDIA SM61 + NVMe SSD):

- Alinhamento de cache para melhor desempenho da CPU
- Alinhamento de warp para operações GPU
- Considerações de tamanho de página NVMe

## Uso

### Alocação básica de memória
```python
from memory_optimization_systems.buddy_allocator_system.buddy_allocator import OptimizedBuddyAllocator

# Criar um alocador com 1MB de memória total e blocos mínimos de 64 bytes
allocator = OptimizedBuddyAllocator(1024*1024, 64)

# Alocar 1024 bytes
result = allocator.allocate(1024)
if result:
    handle, address = result
    print(f"Alocado bloco com handle {handle} no endereço {address}")
    
    # Desalocar o bloco
    success = allocator.deallocate(handle)
    print(f"Desalocação bem-sucedida: {success}")
```

### Alocação de tensores PyTorch
```python
from memory_optimization_systems.buddy_allocator_system.buddy_allocator import PyTorchBuddyAllocator

# Criar um alocador PyTorch-aware
pt_allocator = PyTorchBuddyAllocator(1024*1024, 256)

# Alocar um tensor de tamanho (100, 100) com tipo float32
tensor_result = pt_allocator.allocate_tensor((100, 100), torch.float32)
if tensor_result:
    handle, tensor = tensor_result
    print(f"Tensor alocado: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Desalocar o tensor
    pt_allocator.deallocate_tensor(handle)
```

### Estatísticas
```python
# Obter estatísticas de desempenho
stats = allocator.get_statistics()
print(f"Fragmentação máxima: {stats['max_fragmentation']}")
print(f"Tempo médio de alocação: {stats['avg_allocation_time_ns']} ns")
```

## Otimizações de Hardware

O sistema inclui otimizações específicas para o hardware alvo:

- **Intel i5-10210U**: Alinhamento de cache de linha (64 bytes) e considerações de L3 cache (6MB)
- **NVIDIA SM61**: Alinhamento de warp (32 threads) para operações GPU
- **NVMe SSD**: Considerações de tamanho de página (4096 bytes) para operações de armazenamento

## Segurança de Thread

Todas as operações são thread-safe graças ao uso de `threading.RLock()`, permitindo uso seguro em ambientes multithread.

## Testes

O sistema inclui um conjunto abrangente de testes que validam:

- Funcionalidade básica de alocação/desalocação
- Coalescing de blocos
- Segurança de thread
- Integração PyTorch
- Otimizações de hardware
- Casos de borda e condições de erro

Execute os testes com:
```bash
python -m pytest memory_optimization_systems/buddy_allocator_system/test_buddy_allocator.py -v
```

## Arquitetura

A implementação utiliza uma árvore binária onde:

- Cada nó representa um bloco de memória de tamanho 2^k
- Blocos podem ser divididos em dois blocos "irmãos" de metade do tamanho
- Blocos irmãos podem ser combinados (coalesced) quando ambos estão livres
- A alocação busca blocos livres do tamanho apropriado, subindo na árvore se necessário
- A desalocação libera o bloco e tenta combinar com irmãos adjacentes

## Licença

Este projeto faz parte do sistema Qwen3-VL e está licenciado conforme o arquivo LICENSE principal do projeto.