# Sistema de Feedback para Ajuste Contínuo de Otimizações

Este documento descreve o sistema centralizado de feedback implementado para ajuste contínuo de otimizações em todos os 4 modelos: GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b e Qwen3-vl-2b.

## Componentes do Sistema

### 1. Controlador de Feedback (`feedback_controller.py`)
- Classe `FeedbackController`: Sistema centralizado que monitora métricas de desempenho
- Armazena histórico de métricas para cada modelo
- Avalia desempenho e determina quando ajustes são necessários
- Aplica estratégias de otimização com base no desempenho observado

### 2. Integração de Feedback (`feedback_integration.py`)
- Classe `FeedbackIntegrationMixin`: Mixin para fácil integração com modelos existentes
- Decorador `@monitor_performance`: Para monitoramento automático de funções
- Função `apply_feedback_to_model`: Decorador de classe para aplicar feedback

## Funcionalidades

### Monitoramento de Métricas
O sistema monitora continuamente:
- **Acurácia**: Taxa de precisão das previsões do modelo
- **Latência**: Tempo de resposta para inferência
- **Throughput**: Quantidade de dados processados por unidade de tempo
- **Uso de Memória**: Consumo de memória RAM/GPU
- **Utilização de GPU**: Percentual de uso da GPU

### Estratégias de Ajuste
O sistema pode aplicar automaticamente as seguintes estratégias com base no desempenho:

#### Aumentar Precisão
- Mudar para `float32` para maior precisão
- Reduzir compressão para melhor acurácia
- Aumentar número de heads de atenção

#### Otimizar para Velocidade
- Mudar para `float16` para maior velocidade
- Habilitar técnicas de compressão
- Reduzir número de heads de atenção
- Ajustar tamanho de batch para melhor latência

#### Melhorar Acurácia
- Aumentar precisão do modelo
- Reduzir compressão
- Aumentar heads de atenção

## Integração com Modelos

Todos os 4 modelos foram atualizados para usar o sistema de feedback:

```python
# Exemplo de como os modelos agora usam o feedback
class GLM47Model(FeedbackIntegrationMixin, nn.Module):
    def __init__(self, config):
        super().__init__(model_id="GLM-4-7")  # Inicializa o feedback
        # ... resto da inicialização
        
    def forward(self, *args, **kwargs):
        start_time = time.time()
        result = self._model(*args, **kwargs)
        end_time = time.time()
        
        # Registra métricas de desempenho
        self.record_performance_metrics(
            accuracy=calculated_accuracy,  # Calculado conforme necessário
            latency=end_time - start_time,
            throughput=tokens_processed / (end_time - start_time)
        )
        return result
```

## Uso Prático

### Registro de Métricas
```python
from src.inference_pio.common.feedback_controller import get_feedback_controller

controller = get_feedback_controller()
metrics = PerformanceMetrics(
    accuracy=0.92,
    latency=0.05,
    throughput=200.0
)
controller.record_metrics("GLM-4-7", metrics)
```

### Recuperação de Métricas
```python
current_metrics = controller.get_current_metrics("GLM-4-7")
historical_metrics = controller.get_historical_metrics("GLM-4-7", count=10)
```

## Benefícios

1. **Centralização**: Um único sistema gerencia o feedback para todos os modelos
2. **Adaptabilidade**: Os modelos podem se adaptar automaticamente às condições de desempenho
3. **Monitoramento Contínuo**: Métricas são coletadas continuamente para tomada de decisão
4. **Otimização Dinâmica**: Ajustes são feitos automaticamente com base no desempenho observado
5. **Manutenibilidade**: Código DRY com mixins e decoradores reutilizáveis

## Arquitetura

```
src/inference_pio/common/
├── feedback_controller.py      # Sistema central de feedback
├── feedback_integration.py     # Integração com modelos
├── __init__.py                 # Exportações
└── tests/
    └── test_feedback_controller.py  # Testes
```

Os modelos existentes foram atualizados para herdar de `FeedbackIntegrationMixin` e implementar os métodos necessários para registro de métricas e aplicação de ajustes.

## Testes

O sistema inclui testes abrangentes para garantir o funcionamento correto do controlador de feedback e sua integração com os modelos.