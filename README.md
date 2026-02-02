# Inference-PIO

Inference-PIO is a modular, high-performance inference system built on a self-contained plugin architecture. Each model is completely independent with its own configuration, tests, and benchmarks. The system supports advanced models like GLM-4.7, Qwen3-VL, and Qwen3-Coder.

## 📚 Documentation

*   **[Getting Started](docs/guides/getting_started.md):** Installation, basic usage, and configuration.
*   **[Creating a Model Plugin](docs/creating_model_plugin_guide.md):** Guide to creating new model plugins.
*   **[Supported Models](docs/api/models.md):** List of models and their capabilities.
*   **[System Architecture](docs/api/architecture.md):** Deep dive into the plugin system and design.
*   **[Advanced Features](docs/api/advanced_features.md):** Multimodal attention, streaming, and NAS.
*   **[Benchmarking](docs/guides/benchmarking.md):** Performance measurement guide.
*   **[Coding Standards](docs/standards/CODING.md):** Code style and naming conventions.
*   **[Docstring Standards](docs/standards/DOCSTRINGS.md):** Documentation format guidelines.
*   **[Comment Standards](docs/standards/COMMENTS.md):** Inline and block comment guidelines.
*   **[Testing Standards](docs/standards/TESTING.md):** Test organization and naming conventions.
*   **[Benchmarking Standards](docs/standards/BENCHMARKS.md):** Performance measurement guidelines.

## 🛠 Project Structure

```
.
├── benchmark_results/              # General benchmark results
│   └── general/                  # Cross-model benchmark data
├── docs/                         # Documentation (Guides, API, Standards)
├── examples/                     # Example usage scripts
├── scripts/                      # Utility scripts
│   ├── benchmarking/            # Scripts for running benchmarks
│   ├── development/             # Development and debugging scripts
│   ├── testing/                 # Scripts for running tests
│   └── utils/                   # General utility scripts
├── src/
│   ├── common/                  # Shared utilities and interfaces
│   ├── configs/                 # Global configuration
│   ├── inference/               # Inference engine components
│   ├── models/                  # Individual self-contained model plugins
│   │   ├── glm_4_7_flash/       # GLM-4.7 Flash model with all components
│   │   │   ├── __init__.py      # Module entry point
│   │   │   ├── config.py        # Model-specific config
│   │   │   ├── model.py         # Core model implementation
│   │   │   ├── plugin.py        # Plugin interface implementation
│   │   │   ├── plugin_manifest.json # Plugin metadata for discovery
│   │   │   ├── architecture/    # Architecture-specific implementations
│   │   │   ├── attention/       # Attention mechanisms
│   │   │   ├── fused_layers/    # Fused layer implementations
│   │   │   ├── kv_cache/        # KV cache management
│   │   │   ├── mlp/             # MLP implementations
│   │   │   ├── rotary_embeddings/ # Rotary embedding implementations
│   │   │   ├── specific_optimizations/ # Model-specific optimizations
│   │   │   ├── configs/         # Configuration files
│   │   │   ├── tests/           # Legacy model-specific tests (deprecated)
│   │   │   ├── benchmarks/      # Model-specific benchmarks
│   │   │   └── README.md        # Model-specific documentation
│   │   ├── qwen3_0_6b/          # Qwen3-0.6B model with all components
│   │   ├── qwen3_4b_instruct_2507/ # Qwen3-4B-Instruct-2507 model with all components
│   │   ├── qwen3_coder_30b/     # Qwen3-Coder-30B model with all components
│   │   └── qwen3_vl_2b/         # Qwen3-VL-2B model with all components
│   ├── plugins/                 # Plugin system infrastructure
│   │   ├── __init__.py          # Plugin system entry point
│   │   ├── base/                # Base plugin interfaces
│   │   ├── cpu/                 # CPU-specific plugins
│   │   ├── intel/               # Intel-specific plugins
│   │   └── manager.py           # Plugin manager implementation
│   ├── utils/                   # Utility functions
│   └── model_factory.py         # Model creation factory
├── tests/                       # Organized test structure
│   ├── models/                  # Model-specific tests (organized by model)
│   │   ├── glm_4_7_flash/       # Tests for GLM-4.7 Flash model
│   │   ├── qwen3_0_6b/          # Tests for Qwen3-0.6B model
│   │   ├── qwen3_4b_instruct_2507/ # Tests for Qwen3-4B-Instruct-2507 model
│   │   ├── qwen3_coder_30b/     # Tests for Qwen3-Coder-30B model
│   │   └── qwen3_vl_2b/         # Tests for Qwen3-VL-2B model
│   ├── unit/                    # General unit tests
│   ├── integration/             # General integration tests
│   └── performance/             # General performance tests
├── benchmarks/                  # General benchmarks
└── dev_tools/                   # Development tools and utilities
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -c "from src.model_factory import create_model; m=create_model('glm_4_7_flash'); m.initialize(); print(m.infer('Hello'))"
```

## 🧩 Plugin Discovery System

The system automatically discovers new plugins through:
1. **Directory scanning**: Looks for model directories in `src/models/`
2. **Manifest files**: Each model has a `plugin_manifest.json` file
3. **Auto-registration**: Plugins are automatically registered without manual imports

## 🏗️ Self-Contained Architecture

Each model plugin is completely independent with its own:
- Configuration files in `configs/`
- Model implementation in `model.py`
- Plugin interface in `plugin.py`
- Tests in `tests/` (organized by model)
- Benchmarks in `benchmarks/` (organized by model)
- Optimization implementations in dedicated subdirectories

This ensures that each model can be developed, tested, and deployed independently.

## 🧪 Testing

The project now uses an organized test structure that mirrors the `src/models` hierarchy:

```
tests/
├── models/                  # Model-specific tests
│   ├── glm_4_7_flash/       # Tests for GLM-4.7 Flash model
│   │   ├── unit/            # Unit tests
│   │   ├── integration/     # Integration tests
│   │   └── performance/     # Performance tests
│   ├── qwen3_0_6b/          # Tests for Qwen3-0.6B model
│   ├── qwen3_4b_instruct_2507/ # Tests for Qwen3-4B-Instruct-2507 model
│   ├── qwen3_coder_30b/     # Tests for Qwen3-Coder-30B model
│   └── qwen3_vl_2b/         # Tests for Qwen3-VL-2B model
├── unit/                    # General unit tests
├── integration/             # General integration tests
└── performance/             # General performance tests
```

To run tests for a specific model:
```bash
pytest tests/models/qwen3_0_6b/
```

To run unit tests for a specific model:
```bash
pytest tests/models/qwen3_0_6b/unit/
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer guidelines.

## 🧪 Testes com Funcionalidades Reais

O projeto inclui uma suite abrangente de testes que utilizam funcionalidades reais em vez de simulações excessivas. Esses testes exercitam os caminhos críticos do sistema com dados e operações reais, mantendo a eficiência enquanto aumentam a fidelidade à realidade.

### Tipos de Testes Reais

- **Testes de Funcionalidade**: Verificam a funcionalidade básica do sistema usando componentes reais
- **Testes de Integração**: Testam a interação entre múltiplos componentes do sistema
- **Testes de Desempenho**: Medem métricas reais de desempenho em vez de simulações
- **Testes Funcionais**: Verificam o comportamento do sistema do ponto de vista do usuário
- **Testes de Regressão**: Garantem que alterações não quebrem funcionalidades existentes

### Execução dos Testes Reais

Para executar todos os testes com funcionalidades reais:

```bash
python run_real_tests.py
```

Ou executar testes específicos:

```bash
# Testes de funcionalidade
python -m pytest test_real_functionality.py -v

# Testes de integração
python -m pytest test_real_integration.py -v

# Testes de desempenho
python -m pytest test_real_performance.py -v

# Testes funcionais
python -m pytest test_real_functional.py -v

# Testes de regressão
python -m pytest test_real_regression.py -v
```

## 🔌 Arquitetura Extensível

O projeto implementa uma arquitetura flexível e extensível para fácil inclusão de novos modelos e tipos de teste/benchmark. Cada modelo/plugin é completamente independente com sua própria configuração, testes e benchmarks.

### Adicionando Novos Modelos

Use o assistente de criação de modelos para gerar automaticamente toda a estrutura necessária:

```bash
python create_model.py --name meu-novo-modelo --description "Descrição do novo modelo"
```

### Adicionando Novos Tipos de Teste

Crie novos tipos de testes com o assistente de criação de testes:

```bash
python create_test_type.py --name tipo-de-teste --description "Descrição do tipo de teste"
```

### Adicionando Novos Tipos de Benchmark

Adicione novos benchmarks com o assistente de criação de benchmarks:

```bash
python create_benchmark_type.py --name tipo-de-benchmark --description "Descrição do tipo de benchmark"
```

Para mais detalhes sobre a arquitetura extensível, consulte [EXTENSIBLE_ARCHITECTURE_README.md](EXTENSIBLE_ARCHITECTURE_README.md) e [MODEL_PLUGIN_ARCHITECTURE.md](MODEL_PLUGIN_ARCHITECTURE.md).
