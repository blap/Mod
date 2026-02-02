# Arquitetura Unificada de Testes

Este diretório contém a arquitetura unificada de testes com fixtures e utilitários compartilhados para o projeto Mod, seguindo os padrões estabelecidos nas diretrizes de desenvolvimento.

## Estrutura

```
tests/
├── shared/                 # Componentes compartilhados de teste
│   ├── fixtures/           # Fixtures reutilizáveis
│   │   ├── __init__.py
│   │   ├── plugin_fixtures.py      # Fixtures originais
│   │   ├── standardized_fixtures.py # Fixtures padronizados
│   │   └── example_fixtures.py     # Exemplos de fixtures
│   ├── utils/              # Utilitários de teste
│   │   ├── __init__.py
│   │   ├── test_utils.py
│   │   └── assertions.py
│   ├── example_usage.py    # Exemplo de uso dos componentes
│   └── __init__.py
├── conftest.py             # Configuração principal do pytest
├── unit/                   # Testes unitários
├── integration/            # Testes de integração
├── models/                 # Testes específicos por modelo
└── ...
```

## Componentes Compartilhados

### Fixtures Disponíveis

#### Fixtures Básicas
- `temp_dir`: Diretório temporário para testes
- `temp_file`: Arquivo temporário dentro do diretório temporário
- `sample_text_data`: Dados de texto de amostra
- `sample_tensor_data`: Dados de tensor de amostra
- `parametrized_tensor_data`: Dados de tensor parametrizados para testes com diferentes tamanhos

#### Fixtures de Configuração
- `sample_config`: Configuração de amostra
- `plugin_config_with_gpu`: Configuração de plugin com configurações de GPU
- `sample_plugin_manifest`: Manifesto de plugin de amostra

#### Fixtures de Objetos Mock
- `mock_torch_model`: Modelo PyTorch mock para testes
- `mock_plugin_dependencies`: Dependências mock para testes de plugin
- `realistic_test_plugin`: Instância realista de plugin para testes
- `mock_plugin_with_error_handling`: Plugin mock com tratamento de erros

#### Fixtures Avançadas
- `complex_test_environment`: Ambiente de teste completo com múltiplos recursos
- `device_and_precision_config`: Configurações parametrizadas de dispositivo e precisão
- `specialized_plugin_metadata`: Metadados de plugin especializados
- `expensive_resource`: Recurso caro inicializado uma vez por sessão
- `mocked_network_operations`: Mocks para operações de rede
- `performance_test_data`: Dados grandes para testes de performance
- `concurrency_test_setup`: Recursos para testes de concorrência
- `validated_test_state`: Estado de teste com validação
- `plugin_factory`: Fábrica para criar múltiplas instâncias de plugin
- `integration_test_components`: Múltiplos componentes para testes de integração

### Utilitários Disponíveis

- Funções para criação de dados de teste
- Funções para manipulação de arquivos e diretórios
- Funções para medição de desempenho
- Funções para formatação de dados

### Asserções Compartilhadas

- `assert_plugin_interface_implemented`: Verifica se um plugin implementa a interface correta
- `assert_tensor_properties`: Verifica propriedades de tensores
- `assert_dict_contains_keys`: Verifica se um dicionário contém chaves específicas
- Outras asserções comuns

## Diretrizes para Fixtures

Consulte o documento [Diretrizes para Fixtures e Setups de Testes](../../docs/test_fixtures_guidelines.md) para obter informações detalhadas sobre:

- Princípios fundamentais para criação de fixtures
- Tipos de fixtures e quando usá-los
- Padrões de nomenclatura
- Estrutura recomendada
- Boas práticas
- Considerações de segurança e performance

## Como Usar

Para usar os componentes compartilhados em seus testes:

```python
import pytest

def test_exemplo(realistic_test_plugin):
    # Use a fixture compartilhada
    realistic_test_plugin.initialize()
    result = realistic_test_plugin.infer("entrada de teste")
    assert "Processed: entrada de teste" == result
    realistic_test_plugin.cleanup()

# Exemplo de uso de fixture parametrizado
@pytest.mark.parametrize("input_data,expected", [
    ("hello", "Processed: hello"),
    ("world", "Processed: world")
])
def test_plugin_with_params(realistic_test_plugin, input_data, expected):
    realistic_test_plugin.initialize()
    result = realistic_test_plugin.infer(input_data)
    assert result == expected
    realistic_test_plugin.cleanup()
```

Os fixtures e utilitários são automaticamente disponibilizados em todos os testes graças ao `conftest.py` principal.

## Benefícios

- **Redução de código duplicado**: Componentes reutilizáveis em múltiplos testes
- **Consistência**: Mesma lógica de teste aplicada uniformemente
- **Manutenibilidade**: Alterações em um único lugar afetam todos os testes
- **Facilidade de uso**: Interfaces padronizadas e bem documentadas
- **Flexibilidade**: Fixtures parametrizados e fábricas para diferentes cenários
- **Performance**: Uso apropriado de escopos para otimizar execução de testes