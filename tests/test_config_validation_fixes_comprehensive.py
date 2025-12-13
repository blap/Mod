"""
Comprehensive test to validate all configuration validation and system integration fixes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qwen3_vl.config.config_validation import ConfigValidator, validate_config_file
from src.qwen3_vl.config.validation_system import ConfigValidator as NewConfigValidator, ValidationResult, ValidationLevel, ConfigValidationManager
from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.factory import ConfigFactory
from src.qwen3_vl.components.system.di_container import DIContainer, create_default_container
from src.qwen3_vl.config.memory_config import MemoryConfig
from src.qwen3_vl.config.attention_config import AttentionConfig
from src.qwen3_vl.config.routing_config import RoutingConfig
from src.qwen3_vl.config.hardware_config import HardwareConfig

def test_config_validation_basic():
    """Test basic configuration validation functionality."""
    config = Qwen3VLConfig()
    validator = ConfigValidator()
    
    # Test valid configuration
    result = validator.validate_config(config, strict=False)
    assert result['valid'] == True
    assert len(result['errors']) == 0
    print("[PASS] Basic config validation works")
    
    # Test validation with invalid values using a mock config
    class MockConfig:
        def __init__(self):
            self.num_hidden_layers = 0  # Invalid value
            self.num_attention_heads = 32
            self.hidden_size = 2048
            self.vocab_size = 152064
            self.max_position_embeddings = 32768
    
    mock_config = MockConfig()
    result = validator.validate_config(mock_config, strict=False)
    assert result['valid'] == False
    assert len(result['errors']) > 0
    print("[PASS] Invalid config validation works")


def test_config_factory_from_dict():
    """Test ConfigFactory from_dict functionality."""
    config_dict = {
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'hidden_size': 2048,
        'memory_config': {
            'use_memory_pooling': True,
            'memory_pool_initial_size': 1024 * 1024 * 256
        }
    }
    
    # This should work without errors
    config = ConfigFactory.from_dict(config_dict)
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32
    assert config.hidden_size == 2048
    assert hasattr(config, 'memory_config')
    assert config.memory_config is not None
    assert config.memory_config.use_memory_pooling == True
    print("[PASS] ConfigFactory from_dict works")


def test_config_factory_from_json_file():
    """Test ConfigFactory from_json_file functionality."""
    import tempfile
    import json
    
    config_dict = {
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'hidden_size': 2048
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        temp_file = f.name
    
    try:
        config = ConfigFactory.from_json_file(temp_file)
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.hidden_size == 2048
        print("[PASS] ConfigFactory from_json_file works")
    finally:
        os.unlink(temp_file)


def test_new_config_validation_system():
    """Test the new comprehensive validation system."""
    config = Qwen3VLConfig()
    validator = NewConfigValidator(ValidationLevel.MODERATE)
    
    result = validator.validate_config(config, strict=False)
    assert isinstance(result, ValidationResult)
    assert result.valid == True
    assert len(result.errors) == 0
    print("[PASS] New config validation system works")


def test_di_container_functionality():
    """Test dependency injection container functionality."""
    config = Qwen3VLConfig()
    container = create_default_container(config)
    
    # Test that container is properly set up
    assert container._config is not None
    assert container._components['main_config'] is not None
    print("[PASS] DI Container functionality works")


def test_config_from_dict_missing_imports():
    """Test that config validation doesn't have import issues."""
    # This test verifies that imports work correctly
    from src.qwen3_vl.config.config_validation import validate_config_file
    print("[PASS] Config validation imports work correctly")


def test_validation_with_context():
    """Test validation with specific context."""
    config = Qwen3VLConfig()
    validator = NewConfigValidator(ValidationLevel.MODERATE)
    manager = ConfigValidationManager(ValidationLevel.MODERATE)

    # Test context-specific validation
    result = manager.validate_config_with_context(config, "model")
    assert hasattr(result, 'valid')
    assert hasattr(result, 'errors')
    assert hasattr(result, 'warnings')
    print("[PASS] Context-specific validation works")


def test_config_compatibility_validation():
    """Test configuration compatibility validation."""
    config1 = Qwen3VLConfig()
    config2 = Qwen3VLConfig()
    
    from src.qwen3_vl.config.validation_system import ConfigValidationManager
    manager = ConfigValidationManager()
    
    result = manager.validate_config_compatibility(config1, config2)
    assert isinstance(result, ValidationResult)
    assert result.valid == True  # Same configs should be compatible
    print("[PASS] Config compatibility validation works")


def test_config_validation_edge_cases():
    """Test configuration validation edge cases."""
    # Create a valid config first
    config = Qwen3VLConfig()
    
    # Now test the validation by directly changing values and testing the validation functions
    # This tests the validation functions without creating an invalid Qwen3VLConfig object
    validator = ConfigValidator()
    
    # Create a mock config object with invalid values to test validation functions
    class MockConfig:
        def __init__(self):
            self.num_hidden_layers = 16  # Should be 32
            self.num_attention_heads = 16  # Should be 32
            self.sparsity_ratio = 1.5  # Invalid range
            self.attention_dropout_prob = -0.1  # Invalid range
            self.hidden_size = 2048
            self.vocab_size = 152064
            self.max_position_embeddings = 32768
    
    mock_config = MockConfig()
    result = validator.validate_config(mock_config, strict=False)
    
    # Should have errors for capacity preservation and invalid ranges
    assert result['valid'] == False
    assert len(result['errors']) > 0
    # Check for specific errors
    error_messages = ' '.join(result['errors'])
    assert 'capacity' in error_messages.lower() or '32' in error_messages
    print("[PASS] Config validation edge cases work")


def test_modular_config_imports():
    """Test that all modular config imports work."""
    memory_config = MemoryConfig()
    attention_config = AttentionConfig()
    routing_config = RoutingConfig()
    hardware_config = HardwareConfig()
    
    assert memory_config is not None
    assert attention_config is not None
    assert routing_config is not None
    assert hardware_config is not None
    print("[PASS] Modular config imports work")


def run_all_tests():
    """Run all tests to validate fixes."""
    print("Running comprehensive tests for configuration validation and system integration fixes...")
    
    test_config_validation_basic()
    test_config_factory_from_dict()
    test_config_factory_from_json_file()
    test_new_config_validation_system()
    test_di_container_functionality()
    test_config_from_dict_missing_imports()
    test_validation_with_context()
    test_config_compatibility_validation()
    test_config_validation_edge_cases()
    test_modular_config_imports()
    
    print("\n[ALL TESTS PASSED] Configuration validation and system integration issues have been fixed.")


if __name__ == "__main__":
    run_all_tests()