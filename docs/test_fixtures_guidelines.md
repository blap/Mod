# Diretrizes para Fixtures e Setups de Testes

Este documento estabelece padrões e melhores práticas para a criação e uso de fixtures e setups de testes no projeto Mod.

## Visão Geral

Fixtures são funções especiais do pytest que fornecem dados ou objetos necessários para os testes. Elas ajudam a eliminar código duplicado e garantem consistência nos testes.

## Princípios Fundamentais

1. **Reutilização**: Os fixtures devem ser projetados para serem reutilizáveis em múltiplos testes
2. **Isolamento**: Cada teste deve ser independente e não afetar outros testes
3. **Limpeza**: Todo fixture deve cuidar da limpeza dos recursos que criou
4. **Clareza**: Os nomes dos fixtures devem ser descritivos e autoexplicativos
5. **Eficiência**: Os fixtures devem ser leves e rápidos de executar

## Tipos de Fixtures

### Fixtures de Recursos Temporários
- **Objetivo**: Prover recursos temporários como diretórios, arquivos ou conexões
- **Exemplo**: `temp_dir` fixture que cria e limpa um diretório temporário

### Fixtures de Dados de Teste
- **Objetivo**: Prover dados consistentes para testes
- **Exemplos**: `sample_text_data`, `sample_tensor_data`, `sample_config`

### Fixtures de Objetos Mock
- **Objetivo**: Prover instâncias mockadas de objetos complexos
- **Exemplos**: `mock_torch_model`, `realistic_test_plugin`

### Fixtures de Configuração
- **Objetivo**: Prover configurações padrão para testes
- **Exemplos**: `sample_config`, `sample_plugin_manifest`

## Padrões de Nomenclatura

- Use nomes descritivos e em snake_case
- Prefira nomes curtos mas significativos
- Evite abreviações desnecessárias
- Use prefixos quando apropriado (ex: `mock_`, `sample_`, `temp_`)

## Estrutura Recomendada para Fixtures

```python
import pytest
from pathlib import Path
from typing import Generator

@pytest.fixture
def fixture_name() -> ReturnType:
    """
    Breve descrição do que este fixture faz.
    
    Returns:
        Descrição do tipo retornado
    """
    # Setup: código para preparar o recurso
    resource = create_resource()
    
    # Yield: retorna o recurso para os testes
    yield resource
    
    # Teardown: código para limpar o recurso
    cleanup_resource(resource)
```

## Boas Práticas

### 1. Use Type Hints
Sempre especifique os tipos de retorno dos fixtures para melhor legibilidade e verificação estática:

```python
@pytest.fixture
def sample_tensor_data() -> torch.Tensor:
    return torch.randn(4, 10, 128)
```

### 2. Documente Adequadamente
Inclua docstrings claras explicando o propósito do fixture e seu retorno:

```python
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing and clean it up afterwards.

    Yields:
        Path object to the temporary directory
    """
    temp_path = create_temp_directory()
    yield temp_path
    cleanup_temp_directory(temp_path)
```

### 3. Evite Side Effects
Os fixtures devem ter efeitos colaterais mínimos e previsíveis:

```python
# Bom: fixture puro que apenas retorna dados
@pytest.fixture
def sample_config():
    return {
        "model_path": "/tmp/test_model",
        "batch_size": 4,
        "device": "cpu"
    }

# Ruim: fixture que modifica estado global
@pytest.fixture
def sample_config():
    global_config.model_path = "/tmp/test_model"  # Evite isso!
    return global_config
```

### 4. Use Escopo Adequado
Defina o escopo correto para otimizar o desempenho:

```python
# Para recursos caros de inicializar, use escopo de módulo
@pytest.fixture(scope="module")
def expensive_model():
    return load_heavy_model()

# Para recursos que mudam entre testes, use escopo de função (padrão)
@pytest.fixture
def fresh_database_connection():
    return create_connection()
```

### 5. Encapsule Lógica Complexa
Quando a lógica de setup for complexa, encapsule-a em funções auxiliares:

```python
def create_test_plugin_with_deps():
    """Função auxiliar para criar plugin com dependências."""
    deps = {"torch": Mock(), "numpy": Mock()}
    plugin = TestPlugin(dependencies=deps)
    return plugin

@pytest.fixture
def test_plugin_with_dependencies():
    """Fixture que usa função auxiliar."""
    return create_test_plugin_with_deps()
```

## Organização de Fixtures

### Fixtures Compartilhados
- Localização: `tests/shared/fixtures/`
- Devem ser genéricos o suficiente para serem usados em múltiplos contextos
- Devem seguir os princípios de reutilização e baixo acoplamento

### Fixtures Específicos de Domínio
- Localização: Dentro do diretório de teste correspondente (ex: `tests/models/model_name/`)
- Podem ter dependências específicas do domínio
- Devem ser claramente identificados como específicos

### Hierarquia de Fixtures
- Fixtures mais genéricos devem poder ser usados como base para fixtures mais específicos
- Evite dependências circulares entre fixtures

## Exemplos de Fixtures Bem Estruturados

### Fixture Simples de Dados
```python
@pytest.fixture
def sample_text_data() -> list:
    """
    Provide sample text data for testing.

    Returns:
        List of sample text strings
    """
    return ["text1", "text2", "text3", "text4", "text5"]
```

### Fixture com Setup e Teardown
```python
@pytest.fixture
def temp_database() -> Generator[DatabaseConnection, None, None]:
    """
    Create a temporary in-memory database for testing.

    Yields:
        Database connection object
    """
    db = create_in_memory_db()
    db.connect()
    
    yield db
    
    db.disconnect()
    cleanup_temp_db(db)
```

### Fixture com Parâmetros
```python
@pytest.fixture(params=[1, 2, 4, 8])
def batch_size(request) -> int:
    """
    Parameterized fixture for different batch sizes.
    
    Returns:
        Batch size value
    """
    return request.param
```

## Considerações de Segurança

- Não armazene credenciais ou informações sensíveis em fixtures
- Limpe adequadamente todos os recursos temporários
- Evite fixtures que expõem caminhos de sistema ou informações de ambiente

## Performance

- Minimize o uso de fixtures caros; use escopo adequado
- Considere fixtures lazy (que só criam recursos quando realmente necessários)
- Evite fixtures que realizam operações de E/S desnecessárias

## Testabilidade

- Os fixtures devem ser fáceis de testar individualmente
- Evite fixtures com lógica complexa que precise de seus próprios testes
- Documente bem o comportamento esperado para facilitar a depuração