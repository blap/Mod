# Gerenciadores do Projeto Mod

Este diretório contém implementações concretas das interfaces especializadas para diferentes tipos de funcionalidades no projeto Mod.

## Visão Geral

Os gerenciadores fornecem implementações prontas para uso das interfaces definidas no diretório de interfaces. Eles encapsulam a lógica específica para cada tipo de funcionalidade.

## Tipos de Gerenciadores

### 1. MemoryManager
Implementação concreta de `MemoryManagerInterface`, responsável pelo gerenciamento de memória, incluindo paginação de tensores, swap inteligente, estatísticas de memória e limpeza de recursos.

### 2. DistributedExecutionManager
Implementação concreta de `DistributedExecutionManagerInterface`, que gerencia a execução distribuída e simulação de execução multi-GPU, particionamento de modelo e sincronização.

### 3. TensorCompressionManager
Implementação concreta de `TensorCompressionManagerInterface`, que lida com compressão de tensores, pesos do modelo e ativações, incluindo estatísticas de compressão.

## Uso

Para usar um gerenciador, basta instanciá-lo e chamar seus métodos:

```python
from src.common.managers.memory_manager import MemoryManager

# Criar uma instância do gerenciador de memória
memory_manager = MemoryManager()

# Usar os métodos do gerenciador
success = memory_manager.setup_memory_management()
stats = memory_manager.get_memory_stats()
```

## Benefícios

- **Reutilização**: Implementações prontas para uso em diferentes partes do sistema
- **Padronização**: Implementações consistentes das interfaces
- **Manutenibilidade**: Lógica centralizada em classes específicas
- **Extensibilidade**: Facilidade para estender ou substituir implementações