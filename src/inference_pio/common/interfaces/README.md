# Interfaces do Projeto Mod

Este diretório contém todas as interfaces especializadas para diferentes tipos de funcionalidades no projeto Mod.

## Visão Geral

As interfaces são projetadas para promover a separação de responsabilidades e facilitar a manutenção do código. Cada interface representa um conceito distinto no sistema.

## Tipos de Interfaces

### 1. MemoryManagerInterface
Interface para operações de gerenciamento de memória, incluindo paginação de tensores, swap inteligente e estatísticas de memória.

### 2. DistributedExecutionManagerInterface
Interface para operações de execução distribuída, incluindo simulação de execução multi-GPU e particionamento de modelo.

### 3. TensorCompressionManagerInterface
Interface para operações de compressão de tensores, incluindo compressão de pesos e ativações do modelo.

### 4. SecurityManagerInterface
Interface para operações de segurança, incluindo validação de acesso a arquivos e rede.

### 5. KernelFusionManagerInterface
Interface para operações de fusão de kernels e otimizações de operações do modelo.

### 6. AdaptiveBatchingManagerInterface
Interface para operações de dimensionamento adaptativo de batches com base em métricas de desempenho.

### 7. ModelSurgeryManagerInterface
Interface para operações de cirurgia de modelo para identificar e remover componentes não essenciais.

### 8. PipelineManagerInterface
Interface para operações de pipeline baseados em disco para inferência.

### 9. ShardingManagerInterface
Interface para operações de fragmentação extrema do modelo em centenas de pequenos fragmentos.

## Uso

Para usar uma interface, basta importá-la e implementá-la em sua classe:

```python
from src.common.interfaces.memory_interface import MemoryManagerInterface

class MyMemoryManager(MemoryManagerInterface):
    def setup_memory_management(self, **kwargs) -> bool:
        # Implementação específica
        pass
    
    # Implementar outros métodos obrigatórios...
```

## Benefícios

- **Clareza**: Cada interface tem uma responsabilidade bem definida
- **Flexibilidade**: Facilita a troca de implementações
- **Testabilidade**: Interfaces facilitam a criação de mocks para testes
- **Manutenibilidade**: Mudanças em uma parte do sistema não afetam outras