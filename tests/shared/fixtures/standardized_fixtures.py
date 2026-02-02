"""
Módulo de Fixtures Padrão para Testes

Este módulo contém fixtures pytest padronizados que seguem as melhores práticas
definidas nas diretrizes do projeto Mod. Os fixtures são organizados por categoria
e seguem padrões consistentes de nomenclatura, documentação e tipagem.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock

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
# FIXTURES DE RECURSOS TEMPORÁRIOS
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Cria um diretório temporário para testes e faz a limpeza após o uso.

    Este fixture é útil para testes que precisam de um espaço de arquivo temporário
    para escrita e leitura de dados de teste.

    Yields:
        Path: Objeto de caminho para o diretório temporário
    """
    temp_path = create_temp_directory()
    yield temp_path
    cleanup_temp_directory(temp_path)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Cria um arquivo temporário para testes.

    Este fixture depende do fixture temp_dir e cria um arquivo dentro
    do diretório temporário.

    Args:
        temp_dir: Diretório temporário fornecido pelo fixture correspondente

    Yields:
        Path: Caminho para o arquivo temporário
    """
    temp_file_path = temp_dir / "test_file.tmp"
    temp_file_path.touch()  # Cria o arquivo vazio

    yield temp_file_path

    # O arquivo será removido quando o diretório pai for limpo pelo fixture temp_dir


# =============================================================================
# FIXTURES DE DADOS DE TESTE
# =============================================================================


@pytest.fixture
def sample_text_data() -> List[str]:
    """
    Fornece dados de texto de amostra para testes.

    Retorna uma lista de strings de texto que podem ser usadas para testar
    funcionalidades que lidam com processamento de texto.

    Returns:
        Lista de strings de texto de amostra
    """
    return create_sample_text_data(num_samples=5, max_length=20)


@pytest.fixture
def sample_tensor_data() -> torch.Tensor:
    """
    Fornece dados de tensor de amostra para testes.

    Retorna um tensor PyTorch com dimensões padrão que podem ser usados
    para testar funcionalidades que lidam com operações de tensor.

    Returns:
        Tensor PyTorch de amostra com forma [4, 10, 128]
    """
    return create_sample_tensor_data(batch_size=4, seq_len=10, hidden_size=128)


@pytest.fixture(
    params=[
        {"batch_size": 1, "seq_len": 8, "hidden_size": 64},
        {"batch_size": 2, "seq_len": 16, "hidden_size": 128},
        {"batch_size": 4, "seq_len": 32, "hidden_size": 256},
    ]
)
def parametrized_tensor_data(request) -> torch.Tensor:
    """
    Fornece dados de tensor parametrizados para testes com diferentes tamanhos.

    Este fixture permite testar funcionalidades com diferentes formas de tensor
    usando a funcionalidade de parametrização do pytest.

    Args:
        request: Objeto de solicitação do pytest contendo os parâmetros

    Returns:
        Tensor PyTorch com as dimensões especificadas nos parâmetros
    """
    params = request.param
    return create_sample_tensor_data(
        batch_size=params["batch_size"],
        seq_len=params["seq_len"],
        hidden_size=params["hidden_size"],
    )


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Fornece uma configuração de amostra para testes.

    Retorna um dicionário com configurações padrão que podem ser usadas
    para testar componentes que requerem configuração.

    Returns:
        Dicionário com configurações de amostra
    """
    return {
        "model_path": "/tmp/test_model",
        "batch_size": 4,
        "max_seq_len": 512,
        "device": "cpu",
        "precision": "fp32",
        "use_flash_attention": False,
        "use_quantization": False,
        "num_workers": 2,
        "test_mode": True,
    }


# =============================================================================
# FIXTURES DE OBJETOS MOCK
# =============================================================================


@pytest.fixture
def mock_torch_model() -> torch.nn.Module:
    """
    Fornece um modelo PyTorch mock para testes.

    Retorna uma instância simples de modelo PyTorch que pode ser usada
    para testar funcionalidades que requerem um modelo.

    Returns:
        Instância de modelo PyTorch simples
    """
    return create_mock_model(input_dim=10, output_dim=1)


@pytest.fixture
def mock_plugin_dependencies() -> Dict[str, Any]:
    """
    Fornece dependências mock para testes de plugin.

    Retorna um dicionário com objetos mock que representam dependências
    comuns de plugins, útil para testes de unidade.

    Returns:
        Dicionário com objetos mock representando dependências
    """
    return {
        "torch": Mock(),
        "transformers": Mock(),
        "numpy": Mock(),
        "json": Mock(),
        "datetime": Mock(),
    }


# =============================================================================
# FIXTURES ESPECÍFICOS PARA PLUGINS
# =============================================================================


@pytest.fixture
def sample_metadata() -> PluginMetadata:
    """
    Fornece metadados de amostra para testes de plugin.

    Retorna uma instância de PluginMetadata com dados de amostra
    que podem ser usados para testar funcionalidades de plugin.

    Returns:
        Instância de PluginMetadata com dados de amostra
    """
    return PluginMetadata(
        name="SamplePlugin",
        version="1.0.0",
        author="Sample Author",
        description="Sample Description",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def realistic_test_plugin(sample_metadata: PluginMetadata) -> TextModelPluginInterface:
    """
    Fornece uma instância realista de plugin de teste.

    Retorna uma implementação completa da interface de plugin que pode
    ser usada para testes funcionais e de integração.

    Args:
        sample_metadata: Metadados para usar no plugin

    Returns:
        Instância de plugin de teste realista
    """

    class RealisticTestPlugin(TextModelPluginInterface):
        """Implementação realista de plugin para testes reutilizável."""

        def __init__(
            self, name: str = "TestPlugin", metadata: Optional[PluginMetadata] = None
        ):
            if metadata is None:
                metadata = sample_metadata
                # Criar nova instância com nome específico para evitar colisões
                metadata = PluginMetadata(
                    name=name,
                    version=metadata.version,
                    author=metadata.author,
                    description=metadata.description,
                    plugin_type=metadata.plugin_type,
                    dependencies=metadata.dependencies,
                    compatibility=metadata.compatibility,
                    created_at=metadata.created_at,
                    updated_at=metadata.updated_at,
                )
            super().__init__(metadata)
            self._initialized = False
            self._model = None

        def initialize(self, **kwargs) -> bool:
            """Inicializa o plugin."""
            self._initialized = True
            return True

        def load_model(self, config=None):
            """Carrega o modelo do plugin."""

            # Cria um modelo PyTorch simples
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)

                def forward(self, x):
                    return self.linear(x)

            self._model = SimpleModel()
            return self._model

        def infer(self, data):
            """Executa inferência com o modelo."""
            if not self._initialized:
                self.initialize()

            if isinstance(data, str):
                return f"Processed: {data}"
            elif isinstance(data, (list, tuple)):
                return [f"Processed: {item}" for item in data]
            else:
                return f"Processed: {str(data)}"

        def cleanup(self) -> bool:
            """Limpa os recursos do plugin."""
            self._model = None
            self._initialized = False
            return True

        def supports_config(self, config) -> bool:
            """Verifica se o plugin suporta a configuração fornecida."""
            return config is None or isinstance(config, dict)

        def tokenize(self, text: str, **kwargs):
            """Tokeniza o texto."""
            if not isinstance(text, str):
                raise TypeError("Text must be a string")
            tokens = text.split()
            token_map = {word: idx + 1 for idx, word in enumerate(set(tokens))}
            return [token_map[word] for word in tokens]

        def detokenize(self, token_ids, **kwargs) -> str:
            """Detokeniza IDs de token."""
            if isinstance(token_ids, (list, tuple)):
                return " ".join([f"token_{tid}" for tid in token_ids])
            else:
                return f"token_{token_ids}"

        def generate_text(
            self, prompt: str, max_new_tokens: int = 512, **kwargs
        ) -> str:
            """Gera texto com base no prompt."""
            if not self._initialized:
                self.initialize()
            return f"{prompt} [GENERATED TEXT]"

    return RealisticTestPlugin()


@pytest.fixture
def sample_plugin_manifest() -> Dict[str, Any]:
    """
    Fornece um manifesto de plugin de amostra para testes.

    Retorna um dicionário com informações completas de manifesto
    que podem ser usadas para testar funcionalidades relacionadas a plugins.

    Returns:
        Dicionário com manifesto de plugin de amostra
    """
    return {
        "name": "TestPlugin",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "A test model plugin",
        "plugin_type": "MODEL_COMPONENT",
        "dependencies": ["torch"],
        "compatibility": {"torch_version": ">=2.0.0"},
        "created_at": "2026-01-31T00:00:00",
        "updated_at": "2026-01-31T00:00:00",
        "model_architecture": "TestArch",
        "model_size": "1.0B",
        "required_memory_gb": 1.0,
        "supported_modalities": ["text"],
        "license": "MIT",
        "tags": ["test", "model"],
        "model_family": "TestFamily",
        "num_parameters": 1000000,
        "test_coverage": 1.0,
        "validation_passed": True,
        "main_class_path": "src.models.test_plugin.plugin.TestPlugin",
        "entry_point": "create_test_plugin",
        "input_types": ["text"],
        "output_types": ["text"],
    }


# =============================================================================
# FIXTURES DE CONFIGURAÇÃO AVANÇADA
# =============================================================================


@pytest.fixture
def plugin_config_with_gpu() -> Dict[str, Any]:
    """
    Fornece uma configuração de plugin com configurações de GPU.

    Retorna uma configuração que simula o uso de GPU, útil para testes
    de configuração e compatibilidade.

    Returns:
        Dicionário com configuração de plugin para GPU
    """
    config = {
        "model_path": "/tmp/test_model_gpu",
        "batch_size": 8,
        "max_seq_len": 1024,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "precision": "fp16",
        "use_flash_attention": True,
        "use_quantization": True,
        "num_workers": 4,
        "test_mode": True,
    }
    return config


@pytest.fixture
def mock_plugin_with_error_handling() -> TextModelPluginInterface:
    """
    Fornece um plugin mock com tratamento de erros para testes.

    Retorna uma implementação de plugin que inclui lógica de tratamento
    de erros, útil para testar robustez e resiliência.

    Returns:
        Instância de plugin com tratamento de erros
    """

    class ErrorHandlingPlugin(TextModelPluginInterface):
        """Plugin de teste com tratamento de erros."""

        def __init__(self, name: str = "ErrorHandlingPlugin"):
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                author="Test Author",
                description="Plugin with error handling",
                plugin_type=PluginType.MODEL_COMPONENT,
                dependencies=["torch"],
                compatibility={"torch_version": ">=2.0.0"},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            super().__init__(metadata)
            self._initialized = False
            self._should_fail_initialize = False
            self._should_fail_infer = False

        def initialize(self, **kwargs) -> bool:
            """Inicializa o plugin com possibilidade de falha para testes."""
            if self._should_fail_initialize:
                raise RuntimeError("Initialization failed intentionally for testing")
            self._initialized = True
            return True

        def load_model(self, config=None):
            """Carrega o modelo do plugin."""
            if not self._initialized:
                self.initialize()

            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)

                def forward(self, x):
                    return self.linear(x)

            return SimpleModel()

        def infer(self, data):
            """Executa inferência com tratamento de erro."""
            if not self._initialized:
                self.initialize()

            if self._should_fail_infer:
                raise ValueError("Inference failed intentionally for testing")

            if isinstance(data, str):
                return f"Processed: {data}"
            elif isinstance(data, (list, tuple)):
                return [f"Processed: {item}" for item in data]
            else:
                return f"Processed: {str(data)}"

        def cleanup(self) -> bool:
            """Limpa os recursos do plugin."""
            self._initialized = False
            return True

        def supports_config(self, config) -> bool:
            """Verifica se o plugin suporta a configuração fornecida."""
            return config is None or isinstance(config, dict)

        def tokenize(self, text: str, **kwargs):
            """Tokeniza o texto."""
            if not isinstance(text, str):
                raise TypeError("Text must be a string")
            tokens = text.split()
            token_map = {word: idx + 1 for idx, word in enumerate(set(tokens))}
            return [token_map[word] for word in tokens]

        def detokenize(self, token_ids, **kwargs) -> str:
            """Detokeniza IDs de token."""
            if isinstance(token_ids, (list, tuple)):
                return " ".join([f"token_{tid}" for tid in token_ids])
            else:
                return f"token_{token_ids}"

        def generate_text(
            self, prompt: str, max_new_tokens: int = 512, **kwargs
        ) -> str:
            """Gera texto com base no prompt."""
            if not self._initialized:
                self.initialize()
            return f"{prompt} [GENERATED TEXT]"

        def set_should_fail_initialize(self, should_fail: bool):
            """Configura se a inicialização deve falhar (para testes)."""
            self._should_fail_initialize = should_fail

        def set_should_fail_infer(self, should_fail: bool):
            """Configura se a inferência deve falhar (para testes)."""
            self._should_fail_infer = should_fail

    return ErrorHandlingPlugin()
