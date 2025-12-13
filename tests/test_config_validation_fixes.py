"""
Comprehensive tests for configuration validation and system integration fixes.
This test suite covers the bugs and issues identified in the configuration validation system.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import json
from dataclasses import asdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual modules
from src.qwen3_vl.config.config_validation import ConfigValidator, ConfigValidationError
from src.qwen3_vl.config.validation_system import ConfigValidator as NewConfigValidator, ValidationResult, ValidationLevel
from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.config.factory import ConfigFactory
from src.qwen3_vl.components.system.di_container import DIContainer, create_default_container


def test_config_validation_basic():
    """Test basic configuration validation functionality."""
    config = Qwen3VLConfig()
    validator = ConfigValidator()
    
    # Test valid configuration
    result = validator.validate_config(config, strict=False)
    assert result['valid'] == True
    assert len(result['errors']) == 0
    
    # Test validation with invalid values
    config_invalid = Qwen3VLConfig(num_hidden_layers=0)  # Invalid value
    result = validator.validate_config(config_invalid, strict=False)
    assert result['valid'] == False
    assert len(result['errors']) > 0


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


def test_config_factory_from_json_file():
    """Test ConfigFactory from_json_file functionality."""
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


def test_di_container_functionality():
    """Test dependency injection container functionality."""
    config = Qwen3VLConfig()
    container = create_default_container(config)
    
    # Test that container is properly set up
    assert container._config is not None
    assert container._components['main_config'] is not None


def test_config_from_dict_missing_imports():
    """Test that config validation doesn't have import issues."""
    # Test the config validation module without causing import errors
    from src.qwen3_vl.config.config_validation import validate_config_file
    
    # Create a temporary config file to test
    config_dict = {
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'hidden_size': 2048
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        temp_file = f.name
    
    try:
        # This should not raise an import error
        with patch('src.qwen3_vl.config.config_validation.Qwen3VLConfig') as mock_config:
            # Mock the from_dict method to avoid import issues
            mock_config.from_dict.return_value = Qwen3VLConfig()
            result = validate_config_file(temp_file, strict=False)
            assert result is not None
    finally:
        os.unlink(temp_file)


def test_validation_with_context():
    """Test validation with specific context."""
    config = Qwen3VLConfig()
    validator = NewConfigValidator(ValidationLevel.MODERATE)
    manager = MagicMock()
    manager.validator = validator
    
    # Test context-specific validation
    errors, warnings = validator._validate_context_specific(config, "model")
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_config_compatibility_validation():
    """Test configuration compatibility validation."""
    config1 = Qwen3VLConfig()
    config2 = Qwen3VLConfig()
    
    from src.qwen3_vl.config.validation_system import ConfigValidationManager
    manager = ConfigValidationManager()
    
    result = manager.validate_config_compatibility(config1, config2)
    assert isinstance(result, ValidationResult)
    assert result.valid == True  # Same configs should be compatible


def test_config_validation_edge_cases():
    """Test configuration validation edge cases."""
    # Test with invalid values
    config = Qwen3VLConfig(
        num_hidden_layers=16,  # Should be 32
        num_attention_heads=16,  # Should be 32
        sparsity_ratio=1.5,  # Invalid range
        attention_dropout_prob=-0.1  # Invalid range
    )
    
    validator = ConfigValidator()
    result = validator.validate_config(config, strict=False)
    
    # Should have errors for capacity preservation and invalid ranges
    assert result['valid'] == False
    assert len(result['errors']) > 0
    # Check for specific errors
    error_messages = ' '.join(result['errors'])
    assert 'capacity' in error_messages.lower() or '32' in error_messages


if __name__ == "__main__":
    test_config_validation_basic()
    test_config_factory_from_dict()
    test_config_factory_from_json_file()
    test_new_config_validation_system()
    test_di_container_functionality()
    test_config_from_dict_missing_imports()
    test_validation_with_context()
    test_config_compatibility_validation()
    test_config_validation_edge_cases()
    print("All tests passed!")