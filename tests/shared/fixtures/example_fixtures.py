"""
Exemplos de Fixtures Bem Estruturados

Este módulo contém exemplos de fixtures bem estruturados para diferentes cenários
de teste no projeto Mod. Cada exemplo demonstra práticas recomendadas e padrões
consistentes para a criação de fixtures eficazes.
"""

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Adiciona o diretório src ao caminho para permitir importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.common.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface,
)
from tests.shared.utils.test_utils import (
    cleanup_temp_directory,
    create_mock_model,
    create_sample_tensor_data,
    create_sample_text_data,
    create_temp_directory,
)

# =============================================================================
# EXEMPLO 1: FIXTURE COM SETUP E TEARDOWN COMPLEXO
# =============================================================================


@pytest.fixture
def complex_test_environment() -> Generator[Dict[str, Any], None, None]:
    """
    Exemplo de fixture com setup e teardown complexos.

    Este fixture demonstra como configurar um ambiente de teste completo
    com múltiplos recursos que precisam ser gerenciados.

    Returns:
        Dicionário contendo todos os recursos do ambiente de teste
    """
    # Setup: criar recursos necessários
    temp_dir = create_temp_directory()

    # Criar arquivos de configuração de teste
    config_file = temp_dir / "config.json"
    config_data = {
        "model_path": str(temp_dir / "model.bin"),
        "device": "cpu",
        "batch_size": 4,
    }
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    # Criar modelo de teste
    test_model = create_mock_model(input_dim=128, output_dim=10)
    model_file = temp_dir / "model.bin"
    torch.save(test_model.state_dict(), model_file)

    # Criar dados de teste
    test_data = create_sample_tensor_data(batch_size=4, seq_len=32, hidden_size=128)

    # Preparar o ambiente
    environment = {
        "temp_dir": temp_dir,
        "config_file": config_file,
        "model_file": model_file,
        "test_model": test_model,
        "test_data": test_data,
        "config_data": config_data,
    }

    yield environment

    # Teardown: limpar todos os recursos
    cleanup_temp_directory(temp_dir)


# =============================================================================
# EXEMPLO 2: FIXTURE PARAMETRIZADO
# =============================================================================


@pytest.fixture(
    params=[
        {"device": "cpu", "precision": "fp32"},
        {"device": "cpu", "precision": "fp16"},
        (
            {"device": "cuda:0", "precision": "fp32"}
            if torch.cuda.is_available()
            else {"device": "cpu", "precision": "fp32"}
        ),
        (
            {"device": "cuda:0", "precision": "fp16"}
            if torch.cuda.is_available()
            else {"device": "cpu", "precision": "fp16"}
        ),
    ]
)
def device_and_precision_config(request) -> Dict[str, str]:
    """
    Exemplo de fixture parametrizado para testar diferentes configurações de dispositivo e precisão.

    Este fixture permite testar o código com diferentes combinações de
    dispositivos e precisão de forma sistemática.

    Args:
        request: Objeto de solicitação do pytest contendo os parâmetros

    Returns:
        Dicionário com configuração de dispositivo e precisão
    """
    return request.param


# =============================================================================
# EXEMPLO 3: FIXTURE COM DEPENDÊNCIAS ENTRE FIXTURES
# =============================================================================


@pytest.fixture
def base_plugin_metadata() -> PluginMetadata:
    """
    Fixture base para metadados de plugin.

    Returns:
        Instância básica de PluginMetadata
    """
    return PluginMetadata(
        name="BaseTestPlugin",
        version="1.0.0",
        author="Test Author",
        description="Base plugin for testing",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def specialized_plugin_metadata(base_plugin_metadata: PluginMetadata) -> PluginMetadata:
    """
    Fixture especializado que depende de outro fixture.

    Demonstração de como um fixture pode depender de outro para extender
    ou modificar os dados.

    Args:
        base_plugin_metadata: Metadados base fornecidos pelo fixture correspondente

    Returns:
        Instância modificada de PluginMetadata
    """
    # Criar nova instância com modificações
    return PluginMetadata(
        name=f"Specialized_{base_plugin_metadata.name}",
        version=base_plugin_metadata.version,
        author=base_plugin_metadata.author,
        description=f"Specialized version of {base_plugin_metadata.description}",
        plugin_type=base_plugin_metadata.plugin_type,
        dependencies=base_plugin_metadata.dependencies,
        compatibility=base_plugin_metadata.compatibility,
        created_at=base_plugin_metadata.created_at,
        updated_at=datetime.now(),  # Atualizar timestamp
    )


# =============================================================================
# EXEMPLO 4: FIXTURE COM ESCOPO ESPECÍFICO
# =============================================================================


@pytest.fixture(
    scope="session"
)  # Escopo de sessão - criado uma vez por sessão de teste
def expensive_resource():
    """
    Fixture com escopo de sessão para recursos caros de inicializar.

    Demonstração de como usar escopo apropriado para otimizar o desempenho
    quando o setup é custoso.

    Returns:
        Recurso caro que é inicializado uma vez por sessão de testes
    """
    print("Initializing expensive resource...")

    # Simular inicialização cara (ex: carregar modelo grande, conectar ao banco de dados, etc.)
    class ExpensiveResource:
        def __init__(self):
            # Simular tempo de inicialização
            time.sleep(0.1)  # Em um caso real, seria uma operação cara
            self.initialized_at = datetime.now()
            self.data = list(range(1000))  # Dados caros de gerar

        def get_data_slice(self, start: int, end: int):
            return self.data[start:end]

    resource = ExpensiveResource()

    yield resource

    print("Cleaning up expensive resource...")
    # Aqui você faria qualquer limpeza necessária


# =============================================================================
# EXEMPLO 5: FIXTURE COM MOKCS E PATCHES
# =============================================================================


@pytest.fixture
def mocked_network_operations():
    """
    Fixture que fornece mocks para operações de rede.

    Demonstração de como usar patches para simular operações externas
    como chamadas de rede, acesso a banco de dados, etc.

    Returns:
        Dicionário contendo mocks para diferentes operações
    """
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "socket.socket"
    ) as mock_socket:

        # Configurar comportamentos padrão dos mocks
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok", "data": []}

        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"status": "created", "id": 123}

        mock_socket.return_value.connect.return_value = None

        mocks = {"get": mock_get, "post": mock_post, "socket": mock_socket}

        yield mocks


# =============================================================================
# EXEMPLO 6: FIXTURE PARA TESTES DE PERFORMANCE
# =============================================================================


@pytest.fixture
def performance_test_data():
    """
    Fixture com dados grandes para testes de performance.

    Demonstração de como criar dados específicos para testes de desempenho
    com tamanhos maiores que o normal.

    Returns:
        Dicionário com dados grandes para testes de performance
    """
    # Criar dados maiores para testes de performance
    large_tensor = create_sample_tensor_data(
        batch_size=32,  # Maior que o normal
        seq_len=512,  # Maior que o normal
        hidden_size=512,  # Maior que o normal
    )

    large_text_data = create_sample_text_data(
        num_samples=100, max_length=100  # Mais amostras  # Textos mais longos
    )

    return {
        "large_tensor": large_tensor,
        "large_text_data": large_text_data,
        "size_info": {
            "tensor_shape": large_tensor.shape,
            "text_count": len(large_text_data),
            "estimated_memory_mb": large_tensor.element_size()
            * large_tensor.nelement()
            / (1024 * 1024),
        },
    }


# =============================================================================
# EXEMPLO 7: FIXTURE PARA TESTES DE CONCORRÊNCIA
# =============================================================================


@pytest.fixture
def concurrency_test_setup():
    """
    Fixture para testes de concorrência/threading.

    Demonstração de como configurar um ambiente para testes de concorrência.

    Returns:
        Dicionário com recursos para testes de concorrência
    """

    # Criar um objeto compartilhado para testes de concorrência
    class SharedCounter:
        def __init__(self):
            self.value = 0
            self.lock = threading.Lock()

        def increment(self):
            with self.lock:
                self.value += 1

        def get_value(self):
            with self.lock:
                return self.value

    shared_counter = SharedCounter()

    # Criar uma fila de eventos para coordenação
    event_queue = []

    setup = {
        "shared_counter": shared_counter,
        "event_queue": event_queue,
        "threads": [],
        "stop_event": threading.Event(),
    }

    yield setup

    # Esperar que todas as threads terminem antes de continuar
    for thread in setup["threads"]:
        thread.join(timeout=1.0)  # Timeout de 1 segundo


# =============================================================================
# EXEMPLO 8: FIXTURE COM VALIDAÇÃO DE ESTADO
# =============================================================================


@pytest.fixture
def validated_test_state():
    """
    Fixture com validação de estado antes e depois do uso.

    Demonstração de como validar o estado do sistema antes e depois
    do uso do fixture para garantir integridade.

    Returns:
        Estado de teste validado
    """
    initial_state = {
        "temp_dirs_count": len(list(Path(tempfile.gettempdir()).glob("test_*"))),
        "memory_before": (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        ),
    }

    # Criar estado de teste
    test_state = {
        "data": create_sample_tensor_data(2, 16, 64),
        "counter": 0,
        "flags": set(),
    }

    yield test_state

    # Validar estado após o uso
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_diff = final_memory - initial_state["memory_before"]

    # Verificar vazamento de memória (opcional, dependendo do teste)
    if memory_diff > 1024 * 1024:  # Mais de 1MB de diferença
        print(f"WARNING: Possible memory leak detected: {memory_diff} bytes allocated")

    # Aqui você poderia adicionar outras verificações de estado


# =============================================================================
# EXEMPLO 9: FIXTURE FÁBRICA (FACTORY FIXTURE)
# =============================================================================


@pytest.fixture
def plugin_factory():
    """
    Fixture fábrica para criar múltiplas instâncias de plugin com diferentes configurações.

    Demonstração de como criar uma fixture que atua como fábrica para
    gerar múltiplas instâncias com diferentes parâmetros.

    Returns:
        Função fábrica para criar instâncias de plugin
    """

    def _create_plugin(
        name: str = "TestPlugin", version: str = "1.0.0", should_initialize: bool = True
    ) -> TextModelPluginInterface:
        """Função interna para criar instâncias de plugin."""

        metadata = PluginMetadata(
            name=name,
            version=version,
            author="Factory Generated",
            description=f"Plugin generated by factory: {name}",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch"],
            compatibility={"torch_version": ">=2.0.0"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        class FactoryGeneratedPlugin(TextModelPluginInterface):
            def __init__(self, metadata, should_initialize=True):
                super().__init__(metadata)
                self._initialized = False
                self._should_initialize = should_initialize
                self._model = None

            def initialize(self, **kwargs) -> bool:
                if self._should_initialize:
                    self._initialized = True
                return self._should_initialize

            def load_model(self, config=None):
                if not self._initialized:
                    self.initialize()

                class SimpleModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = torch.nn.Linear(10, 1)

                    def forward(self, x):
                        return self.linear(x)

                self._model = SimpleModel()
                return self._model

            def infer(self, data):
                if not self._initialized:
                    self.initialize()

                if isinstance(data, str):
                    return f"Processed by {self.metadata.name}: {data}"
                elif isinstance(data, (list, tuple)):
                    return [
                        f"Processed by {self.metadata.name}: {item}" for item in data
                    ]
                else:
                    return f"Processed by {self.metadata.name}: {str(data)}"

            def cleanup(self) -> bool:
                self._model = None
                self._initialized = False
                return True

            def supports_config(self, config) -> bool:
                return config is None or isinstance(config, dict)

            def tokenize(self, text: str, **kwargs):
                if not isinstance(text, str):
                    raise TypeError("Text must be a string")
                tokens = text.split()
                token_map = {word: idx + 1 for idx, word in enumerate(set(tokens))}
                return [token_map[word] for word in tokens]

            def detokenize(self, token_ids, **kwargs) -> str:
                if isinstance(token_ids, (list, tuple)):
                    return " ".join([f"token_{tid}" for tid in token_ids])
                else:
                    return f"token_{token_ids}"

            def generate_text(
                self, prompt: str, max_new_tokens: int = 512, **kwargs
            ) -> str:
                if not self._initialized:
                    self.initialize()
                return f"{prompt} [Generated by {self.metadata.name}]"

        return FactoryGeneratedPlugin(metadata, should_initialize)

    return _create_plugin


# =============================================================================
# EXEMPLO 10: FIXTURE PARA TESTES DE INTEGRAÇÃO COM VÁRIOS COMPONENTES
# =============================================================================


@pytest.fixture
def integration_test_components():
    """
    Fixture para testes de integração com múltiplos componentes.

    Demonstração de como configurar múltiplos componentes para testes
    de integração entre diferentes partes do sistema.

    Returns:
        Dicionário com múltiplos componentes para testes de integração
    """
    # Criar múltiplos plugins para teste de integração
    plugins = []

    # Plugin 1: Processador de texto
    text_processor_meta = PluginMetadata(
        name="TextProcessor",
        version="1.0.0",
        author="Integration Test",
        description="Processes text data",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    class TextProcessorPlugin(TextModelPluginInterface):
        def __init__(self, metadata):
            super().__init__(metadata)
            self._initialized = False

        def initialize(self, **kwargs) -> bool:
            self._initialized = True
            return True

        def load_model(self, config=None):
            if not self._initialized:
                self.initialize()
            return create_mock_model(10, 1)

        def infer(self, data):
            if not self._initialized:
                self.initialize()
            if isinstance(data, str):
                return data.upper()
            return data

        def cleanup(self) -> bool:
            self._initialized = False
            return True

        def supports_config(self, config) -> bool:
            return True

        def tokenize(self, text: str, **kwargs):
            return text.split()

        def detokenize(self, token_ids, **kwargs) -> str:
            if isinstance(token_ids, list):
                return " ".join(str(t) for t in token_ids)
            return str(token_ids)

        def generate_text(
            self, prompt: str, max_new_tokens: int = 512, **kwargs
        ) -> str:
            if not self._initialized:
                self.initialize()
            return f"{prompt} [TEXT PROCESSED]"

    text_plugin = TextProcessorPlugin(text_processor_meta)

    # Plugin 2: Processador de números
    number_processor_meta = PluginMetadata(
        name="NumberProcessor",
        version="1.0.0",
        author="Integration Test",
        description="Processes numerical data",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    class NumberProcessorPlugin(TextModelPluginInterface):
        def __init__(self, metadata):
            super().__init__(metadata)
            self._initialized = False

        def initialize(self, **kwargs) -> bool:
            self._initialized = True
            return True

        def load_model(self, config=None):
            if not self._initialized:
                self.initialize()
            return create_mock_model(1, 1)

        def infer(self, data):
            if not self._initialized:
                self.initialize()
            if isinstance(data, (int, float)):
                return data * 2
            elif isinstance(data, list):
                return [x * 2 if isinstance(x, (int, float)) else x for x in data]
            return data

        def cleanup(self) -> bool:
            self._initialized = False
            return True

        def supports_config(self, config) -> bool:
            return True

        def tokenize(self, text: str, **kwargs):
            # Converter números em string para tokens
            try:
                num = float(text)
                return [int(num)]
            except ValueError:
                return [0]

        def detokenize(self, token_ids, **kwargs) -> str:
            if isinstance(token_ids, list):
                return " ".join(str(t) for t in token_ids)
            return str(token_ids)

        def generate_text(
            self, prompt: str, max_new_tokens: int = 512, **kwargs
        ) -> str:
            if not self._initialized:
                self.initialize()
            return f"{prompt} [NUMBERS DOUBLED]"

    number_plugin = NumberProcessorPlugin(number_processor_meta)

    # Criar dados de teste para integração
    test_data = {
        "text_input": "hello world",
        "number_input": 42,
        "mixed_input": ["hello", 123, "world", 456],
    }

    components = {
        "text_processor": text_plugin,
        "number_processor": number_plugin,
        "test_data": test_data,
        "temp_dir": create_temp_directory(),
    }

    yield components

    # Limpar recursos
    cleanup_temp_directory(components["temp_dir"])
