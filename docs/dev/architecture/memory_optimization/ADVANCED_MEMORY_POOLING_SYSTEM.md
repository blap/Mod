# Sistema Avançado de Memory Pooling para Qwen3-VL

Este projeto implementa um sistema avançado de memory pooling otimizado para o modelo Qwen3-VL, especificamente projetado para o hardware Intel i5-10210U + NVIDIA SM61 + NVMe SSD.

## Características do Sistema

### 1. Pools Especializados
- **KV Cache Pool**: Otimizado para armazenamento de chaves e valores de atenção
- **Image Features Pool**: Especializado para features extraídas de imagens
- **Text Embeddings Pool**: Para embeddings de texto
- **Gradients Pool**: Para armazenamento de gradientes durante o treinamento
- **Activations Pool**: Para ativações intermediárias
- **Parameters Pool**: Para parâmetros do modelo

### 2. Algoritmo Buddy Allocation
- Alocação e desalocação eficiente de memória
- Minimização da fragmentação externa
- Fusão automática de blocos adjacentes livres
- Tempo de operação O(log n) para alocação e desalocação

### 3. Gestão de Fragmentação
- Detecção automática de níveis de fragmentação
- Compactação inteligente baseada em thresholds
- Algoritmos de realocação para minimizar fragmentação

### 4. Cache Hierárquico
- Integração com cache de múltiplos níveis (CPU, GPU, disco)
- Política de substituição LRU otimizada
- Migração inteligente entre níveis de cache

### 5. Otimizações de Hardware
- Ajuste automático de tamanhos de bloco com base nas características do hardware
- Consideração de caches CPU (L1, L2, L3) para otimização
- Utilização eficiente da memória GPU SM61
- Otimizações para leitura/escrita em NVMe SSD

## Arquitetura

### Classes Principais

- `AdvancedMemoryPoolingSystem`: Classe principal que coordena todos os pools
- `MemoryPool`: Implementa um pool especializado para um tipo de tensor
- `BuddyAllocator`: Implementa o algoritmo Buddy Allocation
- `HierarchicalCacheManager`: Gerencia o cache hierárquico
- `HardwareOptimizer`: Aplica otimizações específicas para o hardware

### Tipos de Tensores

- `KV_CACHE`: Para cache de atenção
- `IMAGE_FEATURES`: Para features de imagem
- `TEXT_EMBEDDINGS`: Para embeddings de texto
- `GRADIENTS`: Para gradientes
- `ACTIVATIONS`: Para ativações
- `PARAMETERS`: Para parâmetros do modelo

## Benefícios do Sistema

1. **Redução de Fragmentação**: O algoritmo Buddy Allocation minimiza a fragmentação externa
2. **Eficiência de Alocação**: Alocação e desalocação rápidas com complexidade logarítmica
3. **Otimização de Hardware**: Aproveita ao máximo as características específicas do hardware
4. **Escalabilidade**: Sistema pode ser expandido para outros tipos de tensores
5. **Integração com Cache**: Melhora o desempenho através de cache hierárquico

## Uso

```python
from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType

# Criar sistema de memory pooling
memory_system = AdvancedMemoryPoolingSystem(
    kv_cache_size=1024*1024*1024,      # 1GB
    image_features_size=512*1024*1024, # 512MB
    # ... outros parâmetros
)

# Alocar tensor
block = memory_system.allocate(
    TensorType.KV_CACHE, 
    tamanho_em_bytes, 
    "id_do_tensor"
)

# Desalocar tensor
memory_system.deallocate(TensorType.KV_CACHE, "id_do_tensor")

# Obter estatísticas
stats = memory_system.get_system_stats()
```

## Testes

O sistema inclui testes abrangentes que verificam:
- Funcionalidade básica do Buddy Allocator
- Alocação e desalocação corretas
- Fusão de blocos
- Integração entre componentes
- Desempenho de operações

Execute os testes com:
```bash
python test_advanced_memory_pooling_system.py
```

## Desempenho

O sistema foi otimizado para:
- Baixa latência de alocação/desalocação
- Alta taxa de utilização de memória
- Mínima fragmentação
- Eficiência energética no hardware alvo