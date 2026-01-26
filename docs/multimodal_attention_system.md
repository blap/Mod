# Sistema de Atenção Multimodal para Inference-PIO

Este documento descreve o sistema centralizado de atenção multimodal implementado para os modelos GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b e Qwen3-vl-2b no framework Inference-PIO.

## Visão Geral

O sistema de atenção multimodal implementado fornece mecanismos especializados para processamento cruzado entre diferentes modalidades (texto, imagem, áudio). Ele consiste em componentes centrais localizados em `src/inference_pio/common/multimodal_attention.py` que podem ser integrados a qualquer modelo existente no framework.

## Componentes Principais

### 1. MultimodalCrossAttention
- Mecanismo de atenção que permite interação entre diferentes modalidades
- Suporta atenção cruzada entre texto, imagem e áudio
- Projeta cada modalidade separadamente antes de aplicar a atenção

### 2. ModalitySpecificAttention
- Atenção especializada para modalidades específicas (texto, imagem, áudio)
- Aplica processamentos específicos por modalidade quando apropriado
- Mantém a flexibilidade para diferentes tipos de dados

### 3. MultimodalFusionLayer
- Camada que funde informações de múltiplas modalidades
- Combina mecanismos de atenção cruzada com redes feed-forward
- Aplica normalização e conexões residuais

### 4. AdaptiveMultimodalAttention
- Atenção multimodal adaptativa que ajusta seu comportamento com base nas entradas
- Pode adaptar temperatura e esparsidade com base nas características da entrada
- Melhora a eficiência e qualidade em diferentes cenários

## Integração com Modelos

### GLM-4-7
- Integrado via método `_apply_multimodal_attention()` 
- Configurável através do parâmetro `use_multimodal_attention` na configuração
- Adiciona camada de fusão multimodal como atributo do modelo

### Qwen3-4b-instruct-2507
- Integrado via método `_apply_multimodal_attention()`
- Configurável através do parâmetro `use_multimodal_attention` na configuração
- Suporte para modalidades personalizáveis

### Qwen3-coder-30b
- Integrado via método `_apply_multimodal_attention()`
- Configurável para suportar modalidades de código além de texto e imagem
- Otimizado para tarefas de geração de código multimodal

### Qwen3-vl-2b
- Integrado via método `_apply_multimodal_attention()`
- Especialmente otimizado para modelos visão-linguagem
- Expande capacidades multimodais existentes do modelo

## Uso

Para habilitar a atenção multimodal em qualquer modelo, configure o parâmetro `use_multimodal_attention` como `True` e especifique as modalidades suportadas:

```python
config = YourModelConfig()
config.use_multimodal_attention = True
config.modalities = ['text', 'image', 'audio']  # ou quaisquer modalidades necessárias
config.multimodal_dropout = 0.1  # taxa de dropout opcional
```

## Exemplos

Verifique o exemplo em `examples/multimodal_attention_example.py` para demonstrações completas de uso.

## Testes

- Testes unitários: `src/inference_pio/common/tests/test_multimodal_attention.py`
- Testes de integração: `src/inference_pio/common/tests/test_multimodal_attention_integration.py`

Execute com:
```bash
python -m pytest src/inference_pio/common/tests/test_multimodal_attention*.py -v
```

## Arquitetura

O sistema segue o princípio DRY (Don't Repeat Yourself) implementando a lógica central de atenção multimodal em um único local (`src/inference_pio/common/multimodal_attention.py`) e integrando-a aos modelos existentes sem duplicação de código.

As atualizações foram feitas nos seguintes arquivos:
- `src/inference_pio/common/multimodal_attention.py` - Implementação central
- `src/inference_pio/common/__init__.py` - Exportação dos novos módulos
- `src/inference_pio/models/{glm_4_7,qwen3_4b_instruct_2507,qwen3_coder_30b,qwen3_vl_2b}/model.py` - Integração com os modelos
- `src/inference_pio/common/tests/test_multimodal_attention.py` - Testes unitários
- `src/inference_pio/common/tests/test_multimodal_attention_integration.py` - Testes de integração
- `examples/multimodal_attention_example.py` - Exemplo de uso