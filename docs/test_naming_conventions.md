# Convenções de Nomeação para Testes e Benchmarks

Este documento define as convenções de nomeação para classes e métodos de teste e benchmark no projeto Mod.

## Classes de Teste/Benchmark

### Nomeação de Classes
- Usar PascalCase
- Nomear com base no componente ou funcionalidade sendo testada
- Incluir o tipo de teste (Benchmark, Test) no nome quando relevante
- Terminar com "Tests" ou "Benchmarks"
- Exemplos:
  - `LLMInferenceSpeedBenchmarks` (correta)
  - `ModelInitializationTests`
  - `PluginLoadingBenchmarks`

## Métodos de Teste

### Nomeação de Métodos
- Usar snake_case
- Começar com "test_"
- Nomear com base na funcionalidade específica sendo testada
- Incluir o nome do modelo/componente sendo testado
- Seguir o padrão: `test_[modelo]_[funcionalidade]`
- Exemplos:
  - `test_qwen3_0_6b_inference_speed` (correta)
  - `test_glm_4_7_flash_functionality` (correta)
  - `test_plugin_initialization_success`

## Estrutura de Arquivos

### Nomeação de Arquivos
- Usar snake_case
- Começar com "test_" ou conter "_test" no nome
- Nomear com base no tipo de testes contidos
- Exemplos:
  - `test_inference_speed.py`
  - `test_model_functionality.py`
  - `test_plugin_management.py`

## Diretórios

### Organização de Diretórios
- Testes unitários: `tests/unit/`
- Testes de integração: `tests/integration/`
- Testes funcionais: `tests/functional/`
- Benchmarks de desempenho: `benchmarks/performance/`
- Benchmarks funcionais: `benchmarks/functional/`
- Benchmarks de integração: `benchmarks/integration/`

## Documentação

### Docstrings
- Cada classe deve ter uma docstring curta explicando seu propósito
- Cada método de teste deve ter uma docstring explicando o que está sendo testado
- Usar docstrings de aspas triplas
- Ser claro e conciso

## Exemplo Completo

```python
"""LLM inference speed benchmarks."""

import unittest

from src.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin


class LLMInferenceSpeedBenchmarks(unittest.TestCase):
    """LLM inference speed benchmarks."""

    def test_qwen3_0_6b_inference_speed(self):
        """Test inference speed for Qwen3-0.6B model."""
        # Código de teste aqui
        pass
```

## Boas Práticas

- Mantenha os testes independentes uns dos outros
- Use fixtures para configuração e limpeza quando apropriado
- Prefira testes pequenos e focados em um único comportamento
- Use nomes descritivos para facilitar a depuração
- Evite lógica complexa nos testes
- Use asserts específicos para facilitar a depuração