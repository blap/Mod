# Documentação da Estrutura de Testes

## Visão Geral

A estrutura de testes do projeto foi reorganizada para espelhar a hierarquia de `src/models`. Esta mudança visa melhorar a organização, manutenibilidade e compreensão dos testes no projeto Inference-PIO.

## Antes da Reorganização

Anteriormente, os testes estavam organizados por tipo em uma estrutura centralizada:

```
tests/
├── unit/
├── integration/
├── performance/
└── ...
```

Com alguns testes específicos localizados dentro das pastas dos modelos em `src/models/nome_do_modelo/tests/`.

## Depois da Reorganização

A nova estrutura organiza os testes por modelo, mantendo a divisão por tipo de teste:

```
tests/
├── models/
│   ├── glm_4_7_flash/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── performance/
│   ├── qwen3_0_6b/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── performance/
│   ├── qwen3_4b_instruct_2507/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── performance/
│   ├── qwen3_coder_30b/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── performance/
│   └── qwen3_vl_2b/
│       ├── unit/
│       ├── integration/
│       ├── performance/
│       ├── multimodal_projector/
│       ├── vision_transformer/
│       └── visual_resource_compression/
├── unit/
├── integration/
└── performance/
```

## Benefícios da Nova Estrutura

1. **Organização por modelo**: Cada modelo tem seus próprios testes agrupados, facilitando a manutenção e compreensão.

2. **Facilidade de navegação**: É mais fácil encontrar testes relacionados a um modelo específico.

3. **Isolamento**: Testes de diferentes modelos estão claramente isolados uns dos outros.

4. **Escalabilidade**: Adicionar testes para novos modelos segue um padrão claro e consistente.

5. **Manutenibilidade**: Mudanças em um modelo específico afetam apenas os testes relevantes.

## Tipos de Testes

- **Unitários (`unit/`)**: Testam componentes individuais de cada modelo de forma isolada.
- **Integração (`integration/`)**: Testam a interação entre diferentes componentes do modelo.
- **Desempenho (`performance/`)**: Avaliam o desempenho do modelo sob diferentes condições.
- **Componentes específicos**: Subpastas adicionais para testar componentes especializados (ex: `multimodal_projector/`).

## Execução de Testes

### Usando pytest diretamente

Para executar todos os testes de um modelo específico:

```bash
pytest tests/models/qwen3_0_6b/
```

Para executar apenas testes unitários de um modelo:

```bash
pytest tests/models/qwen3_0_6b/unit/
```

Para executar testes de múltiplos modelos:

```bash
pytest tests/models/qwen3_0_6b/ tests/models/glm_4_7_flash/
```

### Usando o script auxiliar

O projeto inclui um script para facilitar a execução de testes específicos:

```bash
python scripts/run_model_tests.py qwen3_0_6b --type unit
```

Para ver todos os modelos disponíveis:

```bash
python scripts/run_model_tests.py --list
```

## Considerações Importantes

1. **Importações**: Os caminhos de importação nos arquivos de teste permanecem inalterados, pois os arquivos fonte (`src/models/...`) não foram movidos.

2. **Compatibilidade**: A estrutura anterior de testes gerais (`tests/unit/`, `tests/integration/`, etc.) continua funcionando para testes que não são específicos de modelo.

3. **Novos modelos**: Ao adicionar um novo modelo, crie a estrutura correspondente em `tests/models/nome_do_novo_modelo/` com as subpastas apropriadas.

## Conclusão

Esta reorganização melhora significativamente a estrutura do projeto, alinhando a organização dos testes com a organização dos modelos, facilitando a manutenção e escalabilidade do código.