# Organização dos Componentes de Otimização

Este documento descreve a estrutura organizada dos componentes de otimização para os modelos Qwen3.

## Estrutura de Diretórios

Cada modelo agora tem uma estrutura consistente de diretórios para componentes de otimização:

```
modelo_nome/
├── attention/                 # Implementações de mecanismo de atenção otimizadas
├── fused_layers/             # Camadas fundidas para melhor desempenho
├── kv_cache/                 # Implementações otimizadas de cache K/V
├── linear_optimizations/     # Otimizações para operações lineares
├── rotary_embeddings/        # Implementações otimizadas de embeddings rotativos
├── specific_optimizations/   # Otimizações específicas para o modelo
├── cuda_kernels/             # Kernels CUDA personalizados
├── tensor_parallel/          # Componentes para paralelização de tensores
├── prefix_caching/           # Implementações para cache de prefixos
├── config.py                 # Configurações do modelo
├── model.py                  # Definição da arquitetura do modelo
├── plugin.py                 # Plugin para integração do modelo
└── ...
```

## Modelos Afetados

### 1. qwen3_0_6b
- Localização: `src/inference_pio/models/qwen3_0_6b/`
- Componentes organizados com arquivos específicos em:
  - `attention/`: `__init__.py`
  - `fused_layers/`: `__init__.py`
  - `kv_cache/`: `__init__.py`
  - `rotary_embeddings/`: `__init__.py`
  - `specific_optimizations/`: `kernels.py`, `__init__.py`

### 2. qwen3_coder_next
- Localização: `src/inference_pio/models/qwen3_coder_next/`
- Componentes organizados com arquivos específicos em:
  - `attention/`: `__init__.py`
  - `fused_layers/`: `norm.py`, `__init__.py`
  - `kv_cache/`: `manager.py`, `__init__.py`
  - `rotary_embeddings/`: `__init__.py`
  - `specific_optimizations/`: `hybrid_scheduler.py`, `__init__.py`
  - `cuda_kernels/`: `setup.py`, `__init__.py`

## Benefícios da Nova Estrutura

1. **Organização Clara**: Cada tipo de otimização está em seu próprio diretório
2. **Manutenibilidade**: Mais fácil encontrar e modificar componentes específicos
3. **Reutilização**: Componentes podem ser compartilhados entre modelos quando apropriado
4. **Escalabilidade**: Novos componentes podem ser adicionados de forma sistemática

## Diretórios Adicionais

Além dos componentes principais de otimização, cada modelo também inclui:

- `benchmarks/`: Scripts de benchmarking específicos do modelo
- `tests/`: Testes unitários e de integração
- `configs/`: Configurações específicas do modelo (quando aplicável)

Esta estrutura padronizada facilita a manutenção e extensão dos modelos Qwen3 com componentes de otimização eficientes.