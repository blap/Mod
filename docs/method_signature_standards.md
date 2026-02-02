# Especificação de Assinaturas de Método para Testes e Benchmarks

Esta documentação define as assinaturas de método padronizadas para testes e benchmarks no projeto Mod.

## Assinaturas de Métodos de Teste Unitário

### Métodos de Inicialização e Limpeza
- `setUp(self)` - Preparação antes de cada teste (unittest)
- `tearDown(self)` - Limpeza após cada teste (unittest)
- `setUpClass(cls)` - Preparação antes de todos os testes na classe (unittest)
- `tearDownClass(cls)` - Limpeza após todos os testes na classe (unittest)

### Métodos de Teste
- `test_nome_do_teste(self)` - Método de teste individual
  - Parâmetros: somente `self`
  - Retorno: nenhum (usa asserções para verificar resultados)

## Assinaturas de Métodos de Benchmark

### Métodos de Inicialização e Limpeza
- `setUp(self)` - Preparação antes de cada benchmark (unittest)
- `setUpClass(cls)` - Preparação antes de todos os benchmarks na classe (unittest)

### Métodos de Benchmark
- `benchmark_nome_do_benchmark(self)` - Método de benchmark individual
  - Parâmetros: somente `self`
  - Retorno: dicionário com resultados do benchmark

### Métodos de Medição
- `run_benchmark(self, target_function, *args, **kwargs)` - Executa uma função de benchmark
  - Parâmetros: `self`, `target_function`, `*args`, `**kwargs`
  - Retorno: dicionário com métricas de desempenho

## Assinaturas de Funções de Utilitário de Teste

### Funções de Criação de Recursos
- `create_test_model(model_name, config=None)` - Cria uma instância de modelo para teste
  - Parâmetros: `model_name` (str), `config` (dict, opcional)
  - Retorno: instância do modelo

- `create_test_plugin(plugin_type, config=None)` - Cria uma instância de plugin para teste
  - Parâmetros: `plugin_type` (str ou enum), `config` (dict, opcional)
  - Retorno: instância do plugin

### Funções de Validação
- `validate_model_output(output, expected_type=None)` - Valida a saída de um modelo
  - Parâmetros: `output` (any), `expected_type` (type, opcional)
  - Retorno: bool

- `validate_plugin_interface(plugin_instance)` - Valida se um plugin implementa a interface correta
  - Parâmetros: `plugin_instance` (objeto plugin)
  - Retorno: bool

## Assinaturas de Funções de Fixture

### Fixtures de Dados
- `get_sample_text_data(size="small")` - Retorna dados de texto de amostra
  - Parâmetros: `size` (str: "small", "medium", "large")
  - Retorno: str

- `get_sample_tensor_data(shape=(10, 10))` - Retorna dados de tensor de amostra
  - Parâmetros: `shape` (tuple)
  - Retorno: torch.Tensor

## Padrões de Asserções

### Asserções Personalizadas
- `assert_model_initialized(self, model)` - Verifica se um modelo está inicializado
  - Parâmetros: `self`, `model`
  - Retorno: nenhum (lança AssertionError se falhar)

- `assert_plugin_has_method(self, plugin, method_name)` - Verifica se um plugin tem um método específico
  - Parâmetros: `self`, `plugin`, `method_name` (str)
  - Retorno: nenhum (lança AssertionError se falhar)

## Diretrizes Gerais

1. Todos os métodos de teste devem começar com "test_"
2. Todos os métodos de benchmark devem começar com "benchmark_" ou estar claramente identificados como benchmarks
3. Os métodos devem ter nomes descritivos que indiquem claramente sua função
4. Parâmetros desnecessários devem ser evitados
5. Tipagem deve ser usada onde apropriado para melhor clareza
6. Os métodos devem retornar valores consistentes com seu propósito (asserções não retornam, benchmarks retornam resultados, utilitários retornam valores úteis)