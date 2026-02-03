# Relatório de Análise de Refatoração

Esta análise detalha a estrutura atual do projeto e propõe uma reorganização modular para melhorar a manutenibilidade, escalabilidade e clareza.

## Estado Atual
*   **Raiz Dispersa:** Scripts e diretórios de teste (`tests/`, `benchmarks/`, `scripts/`) estão na raiz, misturados com configuração.
*   **`src/common` Sobrecarregado:** O diretório `src/common` contém mais de 60 arquivos misturando lógicas de atenção, hardware, configuração, interfaces, e otimização.
*   **`src/models` Segmentado por Tipo:** Os modelos estão aninhados em `src/models/<tipo>/<nome>`, dificultando a portabilidade e independência.
*   **Testes e Benchmarks Separados:** Testes e benchmarks estão distantes da implementação do modelo (`tests/models/...` e `benchmarks/models/...`).
*   **Duplicação:** `rotary_embeddings.py` existe em `src/common` e `src/utils` com implementações ligeiramente diferentes.

## Melhorias Propostas (Ordem Lógica)

### 1. Consolidação da Arquitetura (`src/inference_pio`)
Criar um pacote principal `inference_pio` para abrigar todo o código fonte, evitando poluição do namespace global e facilitando empacotamento.

*   **Ação:** Mover todo o conteúdo relevante de `src/` para `src/inference_pio/`.

### 2. Independência dos Modelos
Mover cada modelo para ser uma unidade autocontida.

*   **Origem:** `src/models/<tipo>/<modelo>`
*   **Destino:** `src/inference_pio/models/<modelo>/`
*   **Inclusão:** Mover testes (`tests/models/<modelo>`) e benchmarks (`benchmarks/models/<modelo>`) para dentro da pasta do modelo (`.../<modelo>/tests`, `.../<modelo>/benchmarks`).

### 3. Modularização de `src/common`
Dividir `src/common` em subpacotes temáticos dentro de `src/inference_pio/common/`:

*   **`attention/`**: `base_attention.py`, `flash_attention_2.py`, `paged_attention.py`, etc.
*   **`optimization/`**: `quantization.py`, `disk_offloading.py`, `kernel_fusion.py`, etc.
*   **`hardware/`**: `hardware_analyzer.py`, `memory_manager.py`, `virtual_device.py`.
*   **`config/`**: `config_manager.py`, `config_loader.py`.
*   **`interfaces/`**: `base_model.py`, `base_plugin_interface.py`.
*   **`layers/`**: Kernels CUDA, `rotary_embeddings.py` (consolidado), `snn.py`.
*   **`processing/`**: `image_tokenization.py`, `streaming_computation.py`.
*   **`parallel/`**: `pipeline_parallel.py`, `model_sharding.py`.
*   **`security/`**: `security_manager.py`, `plugin_isolation.py`.

### 4. Consolidação de Utilitários
*   **Ação:** Fundir `src/utils` e `src/common` onde houver sobreposição.
*   **Resolução de Conflito:** Usar a versão mais completa de `rotary_embeddings.py` (de `src/utils`) e mover para `common/layers/`.

### 5. Centralização de Scripts e Testes Globais
*   **`src/inference_pio/core/`**: Scripts essenciais como `model_factory.py`, `test_discovery.py`.
*   **`src/inference_pio/utils/`**: Utilitários gerais (`testing_utils.py`, `benchmarking_utils.py`).
*   **`src/inference_pio/benchmarks/`**: Benchmarks genéricos e runners.

### 6. Atualização de Imports
Atualizar todos os arquivos para refletir a nova estrutura (ex: `from src.common import ...` -> `from src.inference_pio.common.attention import ...`).

## Benefícios
*   **Coesão:** Código relacionado fica junto.
*   **Independência:** Cada modelo carrega seus próprios testes e requisitos.
*   **Clareza:** Fácil identificar onde uma funcionalidade reside (ex: otimização vs hardware).
