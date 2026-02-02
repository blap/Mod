# Guia para Criar um Plugin de Modelo para o Inference-PIO

## Visão Geral

Este guia descreve como criar um novo plugin de modelo para o sistema Inference-PIO. Cada modelo é implementado como um plugin completamente independente com sua própria configuração, testes e benchmarks que segue uma interface comum e pode ser descoberto e carregado automaticamente pelo sistema.

## Estrutura de Diretórios

Ao criar um novo modelo, você deve seguir a estrutura de diretórios abaixo:

```
src/
└── models/
    └── nome_do_modelo/
        ├── __init__.py
        ├── config.py
        ├── model.py
        ├── plugin.py
        ├── plugin_manifest.json
        ├── architecture/
        ├── attention/
        ├── fused_layers/
        ├── kv_cache/
        ├── mlp/
        ├── rotary_embeddings/
        ├── specific_optimizations/
        ├── tests/
        └── benchmarks/
```

## Documentação Obrigatória

Ao criar um novo modelo, você deve seguir os padrões de documentação do projeto:

### Docstrings
Todos os arquivos devem seguir os padrões de docstring especificados em [DOCSTRINGS.md](../docs/standards/DOCSTRINGS.md):

- Módulos: Devem incluir docstrings explicando o propósito do módulo
- Classes: Devem incluir docstrings com descrição da classe e responsabilidades
- Métodos/Funções: Devem incluir docstrings com Args, Returns e Raises quando aplicável

### Comentários
Siga os padrões de comentário especificados em [COMMENTS.md](../docs/standards/COMMENTS.md):

- Comentários explicativos para código complexo
- Marcadores TODO para funcionalidades futuras
- Comentários sobre otimizações específicas do modelo

## Passos para Criar um Novo Plugin de Modelo

### 1. Criar a Estrutura de Diretórios

Crie uma pasta para o seu modelo em `src/models/nome_do_modelo/`.

### 2. Criar o Manifesto do Plugin

Crie um arquivo `plugin_manifest.json` com as informações do seu modelo:

```json
{
  "name": "NomeDoModelo",
  "version": "1.0.0",
  "author": "Seu Nome",
  "description": "Descrição do seu modelo",
  "plugin_type": "MODEL_COMPONENT",
  "dependencies": [
    "torch",
    "transformers"
  ],
  "compatibility": {
    "torch_version": ">=2.0.0",
    "transformers_version": ">=4.30.0",
    "python_version": ">=3.8",
    "min_memory_gb": 8.0
  },
  "model_architecture": "Tipo de Arquitetura",
  "model_size": "Tamanho do Modelo",
  "required_memory_gb": 8.0,
  "supported_modalities": ["text"],
  "license": "MIT",
  "tags": ["descricao", "do", "modelo"],
  "model_family": "Familia do Modelo",
  "num_parameters": 0,
  "entry_point": {
    "module": ".plugin",
    "class": "NomeDoModelo_Plugin",
    "factory_function": "create_nome_do_modelo_plugin"
  }
}
```

### 3. Criar a Configuração do Modelo

Crie um arquivo `config.py` que herda de `BaseConfig`:

```python
from dataclasses import dataclass
from typing import Optional
from ...common.model_config_base import BaseConfig

@dataclass
class NomeDoModeloConfig(BaseConfig):
    """
    Configuração para o modelo NomeDoModelo.

    Esta classe define todos os parâmetros necessários para o modelo NomeDoModelo,
    incluindo configurações de otimização de memória, mecanismos de atenção e
    otimizações específicas de hardware.
    """
    # Parâmetros específicos do modelo
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 5504
    vocab_size: int = 152064
    max_position_embeddings: int = 32768

    # Parâmetros específicos do modelo NomeDoModelo
    # Adicione aqui os parâmetros específicos do seu modelo

    def __post_init__(self):
        """
        Ajustes pós-inicialização.

        Realiza ajustes após a inicialização da configuração, chamando
        o método correspondente da classe base.
        """
        super().__post_init__()  # Chama o __post_init__ da classe base
        # Adicione ajustes específicos do modelo NomeDoModelo aqui
```

### 4. Criar a Implementação do Modelo

Crie um arquivo `model.py` com a implementação do modelo:

```python
import logging
from typing import Any
import torch
import torch.nn as nn

from .config import NomeDoModeloConfig

logger = logging.getLogger(__name__)

class NomeDoModeloModel(nn.Module):
    """
    Implementação do modelo NomeDoModelo.

    Esta classe implementa o modelo NomeDoModelo com todas as otimizações integradas.
    """
    def __init__(self, config: NomeDoModeloConfig):
        """
        Inicializa o modelo NomeDoModelo com a configuração fornecida.

        Args:
            config: Configuração do modelo NomeDoModelo.
        """
        super().__init__()
        self.config = config

        # Inicializar componentes do modelo aqui
        # Por exemplo: camadas, embeddings, etc.

    def forward(self, *args, **kwargs):
        """
        Passagem para frente do modelo.

        Args:
            *args: Argumentos posicionais para a passagem para frente.
            **kwargs: Argumentos nomeados para a passagem para frente.

        Returns:
            Saída da passagem para frente do modelo.
        """
        # Implementar a passagem para frente
        pass

def create_nome_do_modelo_model(config: NomeDoModeloConfig) -> NomeDoModeloModel:
    """
    Função fábrica para criar uma instância do modelo NomeDoModelo.

    Args:
        config: Configuração do modelo NomeDoModelo.

    Returns:
        Instância do modelo NomeDoModelo.
    """
    return NomeDoModeloModel(config)
```

### 5. Criar o Plugin

Crie um arquivo `plugin.py` que implementa a interface comum:

```python
import logging
from typing import Any
import torch
import torch.nn as nn

from ...common.base_plugin_interface import TextModelPluginInterface, ModelPluginMetadata, PluginType
from .config import NomeDoModeloConfig
from .model import create_nome_do_modelo_model

logger = logging.getLogger(__name__)

class NomeDoModelo_Plugin(TextModelPluginInterface):
    """
    Plugin para o modelo NomeDoModelo.

    Este plugin implementa a interface comum para modelos de linguagem de texto
    e fornece todas as funcionalidades necessárias para execução do modelo NomeDoModelo.
    """
    def __init__(self):
        """
        Inicializa o plugin NomeDoModelo com metadados apropriados.
        """
        metadata = ModelPluginMetadata(
            name="NomeDoModelo",
            version="1.0.0",
            author="Seu Nome",
            description="Plugin para o modelo NomeDoModelo",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch", "transformers"],
            compatibility={
                "torch_version": ">=2.0.0",
                "transformers_version": ">=4.30.0",
                "python_version": ">=3.8",
                "min_memory_gb": 8.0
            }
        )
        super().__init__(metadata)
        self._config = None
        self._model = None
        self._tokenizer = None

    def initialize(self, **kwargs) -> bool:
        """
        Inicializar o plugin com a configuração fornecida.

        Args:
            **kwargs: Argumentos adicionais para inicialização.

        Returns:
            True se a inicialização foi bem-sucedida, False caso contrário.
        """
        try:
            config_data = kwargs.get('config')
            if config_data:
                if isinstance(config_data, dict):
                    self._config = NomeDoModeloConfig(**config_data)
                else:
                    self._config = config_data
            else:
                self._config = NomeDoModeloConfig()

            logger.info("Inicializando o plugin NomeDoModelo...")
            self._model = create_nome_do_modelo_model(self._config)

            # Carregar o tokenizer se necessário
            # self._tokenizer = ...

            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Falha ao inicializar o plugin NomeDoModelo: {e}")
            return False

    def load_model(self, config: NomeDoModeloConfig = None) -> nn.Module:
        """
        Carregar o modelo NomeDoModelo.

        Args:
            config: Configuração opcional para o modelo.

        Returns:
            Instância do modelo carregado.
        """
        if config:
            self._config = config

        if not self.is_loaded:
            self.initialize(config=self._config)

        return self._model

    def infer(self, data: Any) -> Any:
        """
        Executar inferência com o modelo NomeDoModelo.

        Args:
            data: Dados de entrada para a inferência.

        Returns:
            Resultado da inferência.
        """
        if not self.is_loaded:
            self.initialize()

        # Implementar a inferência
        pass

    def supports_config(self, config: Any) -> bool:
        """
        Verificar se este plugin suporta a configuração fornecida.

        Args:
            config: Configuração a ser verificada.

        Returns:
            True se a configuração é suportada, False caso contrário.
        """
        return isinstance(config, NomeDoModeloConfig)

    def tokenize(self, text: str, **kwargs) -> Any:
        """
        Tokenizar o texto fornecido.

        Args:
            text: Texto a ser tokenizado.
            **kwargs: Argumentos adicionais para tokenização.

        Returns:
            Representação tokenizada do texto.
        """
        # Implementar a tokenização
        pass

    def detokenize(self, token_ids: Any, **kwargs) -> str:
        """
        Converter IDs de token de volta para texto.

        Args:
            token_ids: IDs de token a serem convertidos.
            **kwargs: Argumentos adicionais para detokenização.

        Returns:
            Texto reconstruído a partir dos IDs de token.
        """
        # Implementar a detokenização
        pass

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Gerar texto com base no prompt fornecido.

        Args:
            prompt: Prompt para geração de texto.
            max_new_tokens: Número máximo de tokens a serem gerados.
            **kwargs: Argumentos adicionais para geração.

        Returns:
            Texto gerado pelo modelo.
        """
        # Implementar a geração de texto
        pass

    def cleanup(self) -> bool:
        """
        Limpar recursos utilizados pelo plugin.

        Returns:
            True se a limpeza foi bem-sucedida, False caso contrário.
        """
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        return True

def create_nome_do_modelo_plugin() -> NomeDoModelo_Plugin:
    """
    Função fábrica para criar uma instância do plugin NomeDoModelo.

    Returns:
        Instância do plugin NomeDoModelo.
    """
    return NomeDoModelo_Plugin()
```

### 6. Atualizar o Arquivo __init__.py

Atualize o arquivo `__init__.py` do modelo para exportar os componentes:

```python
"""
NomeDoModelo Model Package

This module provides the entry point for the NomeDoModelo model plugin.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from .config import NomeDoModeloConfig
from .model import NomeDoModeloModel, create_nome_do_modelo_model
from .plugin import NomeDoModelo_Plugin, create_nome_do_modelo_plugin

__all__ = [
    "NomeDoModeloConfig",
    "NomeDoModeloModel",
    "create_nome_do_modelo_model",
    "NomeDoModelo_Plugin",
    "create_nome_do_modelo_plugin"
]
```

### 7. Implementar o Método de Instalação

Certifique-se de que o modelo tenha um método `install()` que prepare as dependências necessárias:

```python
# Em model.py
def install(self):
    """
    Preparar as dependências e configurações necessárias para o modelo NomeDoModelo.

    Este método garante que todos os componentes necessários para o modelo NomeDoModelo
    estejam devidamente instalados e configurados antes da execução.
    """
    import subprocess
    import sys

    dependencies = [
        "torch",
        "transformers",
        # Adicione outras dependências específicas do modelo
    ]

    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

    logger.info("Dependências do modelo NomeDoModelo instaladas com sucesso")
```

### 8. Testar o Plugin

Crie testes para garantir que o plugin funcione corretamente:

```python
# Em tests/test_nome_do_modelo.py
import unittest
from src.models.nome_do_modelo.plugin import create_nome_do_modelo_plugin

class TestNomeDoModelo(unittest.TestCase):
    """
    Testes para o modelo NomeDoModelo.
    """
    def test_create_plugin(self):
        """
        Testa a criação do plugin NomeDoModelo.
        """
        plugin = create_nome_do_modelo_plugin()
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.metadata.name, "NomeDoModelo")

    def test_initialize_plugin(self):
        """
        Testa a inicialização do plugin NomeDoModelo.
        """
        plugin = create_nome_do_modelo_plugin()
        result = plugin.initialize()
        self.assertTrue(result)
```

## Considerações Finais

- Cada modelo deve ser autocontido e não depender de outros modelos
- Siga os padrões de nomenclatura e estrutura de diretórios
- Use os sistemas de configuração e plugin existentes
- Certifique-se de que o modelo tenha valores padrão razoáveis
- Implemente os métodos obrigatórios da interface comum
- Siga os padrões de documentação (docstrings e comentários)
- Teste o plugin para garantir que ele funcione corretamente

Com esta estrutura, o novo modelo será automaticamente descoberto e carregado pelo sistema Inference-PIO sem necessidade de edições manuais em outros arquivos do sistema.