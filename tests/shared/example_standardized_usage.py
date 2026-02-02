"""
Exemplo de Testes Utilizando Fixtures Padronizados

Este arquivo demonstra como utilizar os fixtures padronizados no projeto Mod
para diferentes cenários de teste.
"""

from pathlib import Path

import pytest
import torch


def test_basic_fixtures_usage(
    temp_dir: Path,
    sample_text_data: list,
    sample_tensor_data: torch.Tensor,
    mock_torch_model: torch.nn.Module,
    sample_config: dict,
):
    """
    Teste demonstrando o uso básico dos fixtures padronizados.
    """
    # Verificar fixture temp_dir
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()

    # Criar um arquivo temporário para verificar funcionamento
    test_file = temp_dir / "test.txt"
    test_file.write_text("conteúdo de teste")
    assert test_file.exists()

    # Verificar fixture sample_text_data
    assert isinstance(sample_text_data, list)
    assert len(sample_text_data) == 5
    assert all(isinstance(text, str) for text in sample_text_data)

    # Verificar fixture sample_tensor_data
    assert isinstance(sample_tensor_data, torch.Tensor)
    assert sample_tensor_data.shape[0] == 4  # batch size
    assert sample_tensor_data.shape[1] == 10  # sequence length
    assert sample_tensor_data.shape[2] == 128  # hidden size

    # Verificar fixture mock_torch_model
    assert isinstance(mock_torch_model, torch.nn.Module)

    # Testar o modelo mock com entrada
    input_tensor = torch.randn(1, 10)
    output = mock_torch_model(input_tensor)
    assert output.shape == (1, 1)

    # Verificar fixture sample_config
    assert isinstance(sample_config, dict)
    assert "model_path" in sample_config
    assert "batch_size" in sample_config
    assert "device" in sample_config


def test_plugin_fixtures_usage(realistic_test_plugin):
    """
    Teste demonstrando o uso de fixtures específicos para plugins.
    """
    from tests.shared.utils.assertions import assert_plugin_interface_implemented

    # Verificar que o plugin implementa a interface correta
    assert_plugin_interface_implemented(realistic_test_plugin)

    # Testar funcionalidades do plugin
    assert realistic_test_plugin.initialize() is True

    # Testar inferência com string
    result = realistic_test_plugin.infer("teste de entrada")
    assert result == "Processed: teste de entrada"

    # Testar inferência com lista
    result_list = realistic_test_plugin.infer(["entrada1", "entrada2"])
    assert result_list == ["Processed: entrada1", "Processed: entrada2"]

    # Testar tokenização
    tokens = realistic_test_plugin.tokenize("ola mundo ola")
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Testar geração de texto
    generated = realistic_test_plugin.generate_text("prompt de teste")
    assert "prompt de teste" in generated
    assert "[GENERATED TEXT]" in generated

    # Limpar recursos
    assert realistic_test_plugin.cleanup() is True


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_size",
    [
        (1, 8, 64),
        (2, 16, 128),
        (4, 32, 256),
    ],
)
def test_parametrized_tensor_fixture(batch_size, seq_len, hidden_size):
    """
    Teste demonstrando o uso de fixture parametrizado.
    """
    # Este teste usa o fixture parametrized_tensor_data implicitamente
    # através da parametrização manual
    tensor = torch.randn(batch_size, seq_len, hidden_size)
    assert tensor.shape == (batch_size, seq_len, hidden_size)


def test_device_and_precision_fixture(device_and_precision_config: dict):
    """
    Teste demonstrando o uso de fixture parametrizado para configurações de dispositivo.
    """
    assert isinstance(device_and_precision_config, dict)
    assert "device" in device_and_precision_config
    assert "precision" in device_and_precision_config

    device = device_and_precision_config["device"]
    precision = device_and_precision_config["precision"]

    # Verificar que o dispositivo é válido
    assert isinstance(device, str)
    # Verificar que a precisão é válida
    assert isinstance(precision, str)
    assert precision in ["fp32", "fp16", "bf16"]


def test_complex_test_environment(complex_test_environment: dict):
    """
    Teste demonstrando o uso de fixture com setup e teardown complexos.
    """
    env = complex_test_environment

    # Verificar que todos os componentes do ambiente existem
    assert "temp_dir" in env
    assert "config_file" in env
    assert "model_file" in env
    assert "test_model" in env
    assert "test_data" in env
    assert "config_data" in env

    # Verificar que os arquivos existem
    assert env["temp_dir"].exists()
    assert env["config_file"].exists()
    assert env["model_file"].exists()

    # Verificar que os objetos são do tipo correto
    assert isinstance(env["test_model"], torch.nn.Module)
    assert isinstance(env["test_data"], torch.Tensor)


def test_mock_with_error_handling(mock_plugin_with_error_handling):
    """
    Teste demonstrando o uso de fixture com tratamento de erros.
    """
    # Testar comportamento normal
    assert mock_plugin_with_error_handling.initialize() is True

    # Testar inferência normal
    result = mock_plugin_with_error_handling.infer("entrada normal")
    assert "Processed" in result

    # Configurar para falhar na inicialização e testar
    mock_plugin_with_error_handling.set_should_fail_initialize(True)

    with pytest.raises(RuntimeError, match="Initialization failed intentionally"):
        mock_plugin_with_error_handling.initialize()

    # Restaurar comportamento normal
    mock_plugin_with_error_handling.set_should_fail_initialize(False)
    assert mock_plugin_with_error_handling.initialize() is True


def test_performance_test_data(performance_test_data: dict):
    """
    Teste demonstrando o uso de fixture com dados grandes para performance.
    """
    perf_data = performance_test_data

    assert "large_tensor" in perf_data
    assert "large_text_data" in perf_data
    assert "size_info" in perf_data

    # Verificar que o tensor é grande o suficiente
    tensor = perf_data["large_tensor"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 32  # batch size maior
    assert tensor.shape[1] == 512  # sequence length maior
    assert tensor.shape[2] == 512  # hidden size maior

    # Verificar que os dados de texto são grandes o suficiente
    text_data = perf_data["large_text_data"]
    assert isinstance(text_data, list)
    assert len(text_data) == 100  # número maior de amostras

    # Verificar informações de tamanho
    size_info = perf_data["size_info"]
    assert "tensor_shape" in size_info
    assert "text_count" in size_info
    assert "estimated_memory_mb" in size_info


def test_plugin_factory_fixture(plugin_factory):
    """
    Teste demonstrando o uso de fixture fábrica.
    """
    # Criar primeiro plugin
    plugin1 = plugin_factory(name="FirstPlugin", version="1.0.0")
    assert plugin1.metadata.name == "FirstPlugin"
    assert plugin1.metadata.version == "1.0.0"

    # Criar segundo plugin com configurações diferentes
    plugin2 = plugin_factory(
        name="SecondPlugin", version="2.0.0", should_initialize=False
    )
    assert plugin2.metadata.name == "SecondPlugin"
    assert plugin2.metadata.version == "2.0.0"

    # Verificar que o segundo plugin não inicializa
    init_result = plugin2.initialize()
    assert init_result is False  # Porque definimos should_initialize=False


def test_integration_test_components(integration_test_components: dict):
    """
    Teste demonstrando o uso de fixture com múltiplos componentes para integração.
    """
    components = integration_test_components

    # Verificar que todos os componentes existem
    assert "text_processor" in components
    assert "number_processor" in components
    assert "test_data" in components
    assert "temp_dir" in components

    # Testar o processador de texto
    text_plugin = components["text_processor"]
    text_plugin.initialize()
    text_result = text_plugin.infer("hello world")
    assert text_result == "HELLO WORLD"  # Deveria converter para maiúsculas

    # Testar o processador de números
    number_plugin = components["number_processor"]
    number_plugin.initialize()
    number_result = number_plugin.infer(21)
    assert number_result == 42  # Deveria dobrar o valor

    # Verificar dados de teste
    test_data = components["test_data"]
    assert "text_input" in test_data
    assert "number_input" in test_data
    assert "mixed_input" in test_data

    # Verificar diretório temporário
    assert components["temp_dir"].exists()


# Teste adicional para demonstrar uso combinado de múltiplos fixtures
def test_combined_fixtures_usage(
    realistic_test_plugin, sample_config: dict, sample_text_data: list
):
    """
    Teste demonstrando o uso combinado de múltiplos fixtures.
    """
    # Inicializar o plugin com configuração
    plugin = realistic_test_plugin
    plugin.initialize()

    # Usar dados de texto de amostra para testar o plugin
    for text in sample_text_data[:2]:  # Usar apenas os dois primeiros
        result = plugin.infer(text)
        assert "Processed:" in result
        assert text in result

    # Verificar que a configuração tem os campos esperados
    expected_fields = ["model_path", "batch_size", "device"]
    for field in expected_fields:
        assert field in sample_config

    # Limpar recursos
    plugin.cleanup()
