"""
Consolidated tests for plugin interface compliance and functionality.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest
import torch

from src.inference_pio.common.interfaces.base_plugin_interface import (
    ModelPluginInterface as BaseModelPluginInterface,
)
from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface,
    PluginMetadata,
    PluginType,
    StandardPluginInterface,
    TextModelPluginInterface,
)
from tests.conftest import assert_plugin_interface_implemented, realistic_test_plugin


def test_standard_plugin_interface_abstract():
    """Test that StandardPluginInterface is properly abstract."""
    with pytest.raises(TypeError):

        class ConcreteStandardPlugin(StandardPluginInterface):
            raise NotImplementedError("Method not implemented")

        ConcreteStandardPlugin(Mock())


def test_model_plugin_interface_abstract():
    """Test that ModelPluginInterface is properly abstract."""
    with pytest.raises(TypeError):

        class ConcreteModelPlugin(ModelPluginInterface):
            raise NotImplementedError("Method not implemented")

        metadata = PluginMetadata(
            name="TestModel",
            version="1.0.0",
            author="Test Author",
            description="Test Description",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        ConcreteModelPlugin(metadata)


def test_text_model_plugin_interface_abstract():
    """Test that TextModelPluginInterface is properly abstract."""
    with pytest.raises(TypeError):

        class ConcreteTextModelPlugin(TextModelPluginInterface):
            raise NotImplementedError("Method not implemented")

        metadata = PluginMetadata(
            name="TestModel",
            version="1.0.0",
            author="Test Author",
            description="Test Description",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        ConcreteTextModelPlugin(metadata)


def test_realistic_plugin_implementation(realistic_test_plugin):
    """Test that the realistic test plugin properly implements the interface."""
    plugin = realistic_test_plugin

    # Verify it implements the interface
    assert_plugin_interface_implemented(plugin)

    # Test that all methods work properly
    assert plugin.initialize() is True
    model = plugin.load_model()
    assert isinstance(model, torch.nn.Module)

    result = plugin.infer("test input")
    assert "Processed: test input" in result

    tokens = plugin.tokenize("hello world test")
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    detokenized = plugin.detokenize(tokens)
    assert isinstance(detokenized, str)

    generated = plugin.generate_text("test prompt")
    assert "[GENERATED TEXT]" in generated

    assert plugin.cleanup() is True


def test_plugin_metadata_creation():
    """Test that PluginMetadata can be created with all required fields."""
    metadata = PluginMetadata(
        name="TestModel",
        version="1.0.0",
        author="Test Author",
        description="Test Description",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch", "transformers"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
        model_architecture="Transformer",
        model_size="7B",
        required_memory_gb=8.0,
        supported_modalities=["text"],
        license="MIT",
        tags=["test", "model"],
        model_family="TestFamily",
        num_parameters=7000000000,
        test_coverage=0.95,
        validation_passed=True,
    )

    assert metadata.name == "TestModel"
    assert metadata.version == "1.0.0"
    assert metadata.plugin_type == PluginType.MODEL_COMPONENT
    assert "torch" in metadata.dependencies
    assert metadata.model_architecture == "Transformer"
    assert metadata.required_memory_gb == 8.0
    assert "text" in metadata.supported_modalities
    assert "test" in metadata.tags
    assert metadata.num_parameters == 7000000000
    assert metadata.test_coverage == 0.95
    assert metadata.validation_passed is True


def test_plugin_type_enum():
    """Test that PluginType enum contains expected values."""
    assert PluginType.MODEL_COMPONENT.value == "model_component"
    assert PluginType.ATTENTION.value == "attention"
    assert PluginType.MEMORY_MANAGER.value == "memory_manager"
    assert PluginType.OPTIMIZATION.value == "optimization"
    assert PluginType.HARDWARE.value == "hardware"
    assert PluginType.PERFORMANCE.value == "performance"
    assert PluginType.TRAINING_STRATEGY.value == "training_strategy"
    assert PluginType.INFERENCE_STRATEGY.value == "inference_strategy"
    assert PluginType.DATA_PROCESSOR.value == "data_processor"
    assert PluginType.METRIC.value == "metric"
    assert PluginType.TUNING_STRATEGY.value == "tuning_strategy"
    assert PluginType.KV_CACHE.value == "kv_cache"


def test_interface_method_signatures(realistic_test_plugin):
    """Test that the realistic plugin has correct method signatures."""
    plugin = realistic_test_plugin

    import inspect

    # Test initialize method signature
    init_sig = inspect.signature(plugin.initialize)
    assert list(init_sig.parameters.keys()) == ["kwargs"]
    assert init_sig.return_annotation == bool

    # Test load_model method signature
    load_sig = inspect.signature(plugin.load_model)
    params = list(load_sig.parameters.keys())
    assert "config" in params

    # Test infer method signature
    infer_sig = inspect.signature(plugin.infer)
    params = list(infer_sig.parameters.keys())
    assert "data" in params

    # Test cleanup method signature
    cleanup_sig = inspect.signature(plugin.cleanup)
    assert len(cleanup_sig.parameters) == 0
    assert cleanup_sig.return_annotation == bool

    # Test supports_config method signature
    supports_sig = inspect.signature(plugin.supports_config)
    params = list(supports_sig.parameters.keys())
    assert "config" in params


def test_text_model_interface_methods(realistic_test_plugin):
    """Test that text model interface methods have correct signatures."""
    plugin = realistic_test_plugin

    import inspect

    # Test tokenize method signature
    tokenize_sig = inspect.signature(plugin.tokenize)
    params = list(tokenize_sig.parameters.keys())
    assert "text" in params

    # Test detokenize method signature
    detokenize_sig = inspect.signature(plugin.detokenize)
    params = list(detokenize_sig.parameters.keys())
    assert "token_ids" in params

    # Test generate_text method signature
    gen_sig = inspect.signature(plugin.generate_text)
    params = list(gen_sig.parameters.keys())
    assert "prompt" in params
    assert "max_new_tokens" in params


def test_plugin_inheritance_chain(realistic_test_plugin):
    """Test that plugins properly inherit from the interface hierarchy."""
    plugin = realistic_test_plugin

    # Verify inheritance chain
    assert isinstance(plugin, TextModelPluginInterface)
    assert isinstance(plugin, ModelPluginInterface)
    assert isinstance(plugin, StandardPluginInterface)


def test_plugin_attributes(realistic_test_plugin):
    """Test that plugins have required attributes."""
    plugin = realistic_test_plugin

    # Test required attributes
    assert hasattr(plugin, "metadata")
    assert hasattr(plugin, "is_loaded")
    assert hasattr(plugin, "is_active")
    assert hasattr(plugin, "_initialized")

    # Test initial state
    assert plugin.is_loaded is False
    assert plugin.is_active is False
    assert plugin._initialized is False


if __name__ == "__main__":
    pytest.main([__file__])
