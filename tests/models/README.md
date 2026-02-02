# Estrutura de Testes por Modelo

Esta pasta contém a estrutura organizada de testes que espelha a hierarquia de `src/models`. Cada modelo tem sua própria pasta de testes com subpastas para diferentes tipos de testes.

## Estrutura

```
tests/
└── models/
    ├── glm_4_7_flash/
    │   ├── unit/
    │   ├── integration/
    │   └── performance/
    ├── qwen3_0_6b/
    │   ├── unit/
    │   ├── integration/
    │   └── performance/
    ├── qwen3_4b_instruct_2507/
    │   ├── unit/
    │   ├── integration/
    │   └── performance/
    ├── qwen3_coder_30b/
    │   ├── unit/
    │   ├── integration/
    │   └── performance/
    └── qwen3_vl_2b/
        ├── unit/
        ├── integration/
        ├── performance/
        ├── multimodal_projector/
        ├── vision_transformer/
        └── visual_resource_compression/
```

## Diretrizes

- Os testes unitários (`unit/`) testam componentes individuais de cada modelo.
- Os testes de integração (`integration/`) testam a interação entre diferentes componentes do modelo.
- Os testes de desempenho (`performance/`) avaliam o desempenho do modelo sob diferentes condições.
- Subpastas específicas (como `multimodal_projector/`) contêm testes para componentes especializados de modelos multimodais.

## Adicionando novos testes

Ao adicionar testes para um novo modelo ou componente, siga esta estrutura e coloque os testes no diretório apropriado.