"""
Test suite for validating capacity preservation under all optimization combinations.
This test suite ensures that the model maintains 32 transformer layers and 32 attention heads
under various optimization configurations.
"""

import sys
import os
# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import torch.nn as nn
import pytest
from typing import Dict, Any, List

# Import the capacity verification module directly
import importlib.util
capacity_verification_spec = importlib.util.spec_from_file_location(
    "capacity_verification",
    os.path.join(os.path.dirname(__file__), "capacity_verification.py")
)
capacity_verification = importlib.util.module_from_spec(capacity_verification_spec)
capacity_verification_spec.loader.exec_module(capacity_verification)

# Import the config module directly
config_spec = importlib.util.spec_from_file_location(
    "config",
    os.path.join(os.path.dirname(__file__), "..", "core", "config.py")
)
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)

# Create aliases for the classes we need
ModelCapacityValidator = capacity_verification.ModelCapacityValidator
OptimizationIntegrationValidator = capacity_verification.OptimizationIntegrationValidator
ContinuousMonitoringSystem = capacity_verification.ContinuousMonitoringSystem
ErrorReportingSystem = capacity_verification.ErrorReportingSystem
ValidationResult = capacity_verification.ValidationResult
ValidationStatus = capacity_verification.ValidationStatus
validate_model_capacity_preservation = capacity_verification.validate_model_capacity_preservation
Qwen3VLConfig = config_module.Qwen3VLConfig


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for testing purposes."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Create a simple attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward
        ff_output = self.mlp(x)
        x = self.layer_norm2(x + ff_output)

        return x


class MockLanguageModel(nn.Module):
    """Mock language model with transformer layers."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MockTransformerLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MockVisionTower(nn.Module):
    """Mock vision tower for multimodal model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create vision transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.vision_hidden_size,
                nhead=config.vision_num_attention_heads,
                batch_first=True
            ) for _ in range(config.vision_num_hidden_layers)
        ])
        self.projection = nn.Linear(config.vision_hidden_size, config.hidden_size)

    def forward(self, pixel_values):
        # Process through vision transformer layers
        batch_size, num_patches, hidden_size = pixel_values.shape
        hidden_states = pixel_values
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Project to language model dimension
        hidden_states = self.projection(hidden_states)
        return hidden_states


class MockQwen3VLModel(nn.Module):
    """Mock Qwen3-VL model for testing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = MockLanguageModel(config)
        self.vision_tower = MockVisionTower(config)
        self.multi_modal_projector = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None):
        outputs = {}
        
        if input_ids is not None:
            lang_outputs = self.language_model(input_ids, attention_mask)
            outputs['language'] = lang_outputs
        
        if pixel_values is not None:
            vision_outputs = self.vision_tower(pixel_values)
            outputs['vision'] = vision_outputs
        
        return outputs


def create_test_model(config: Qwen3VLConfig) -> MockQwen3VLModel:
    """Create a test model with the given configuration."""
    return MockQwen3VLModel(config)


def test_basic_capacity_preservation():
    """Test basic capacity preservation with default configuration."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    # Validate capacity
    results = validate_model_capacity_preservation(model)
    
    assert results['summary']['capacity_preserved'], "Basic model should preserve capacity"
    assert results['summary']['failed'] == 0, "No validation checks should fail for basic model"
    
    print("PASS: Basic capacity preservation test passed")


def test_layer_count_validation():
    """Test layer count validation specifically."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    result = validator.validate_layer_count(model)
    
    assert result.status == ValidationStatus.PASSED, f"Layer count validation failed: {result.message}"
    assert result.details['actual_layers'] == 32, f"Expected 32 layers, got {result.details['actual_layers']}"
    
    print("PASS: Layer count validation test passed")


def test_attention_heads_validation():
    """Test attention heads validation specifically."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    result = validator.validate_attention_heads(model)

    assert result.status == ValidationStatus.PASSED, f"Attention heads validation failed: {result.message}"
    # Check that all found attention heads are 32
    all_heads_correct = all(head_count == 32 for head_count in result.details['actual_heads'])
    assert all_heads_correct, f"Expected all heads to be 32, got {result.details['actual_heads']}"

    print("PASS: Attention heads validation test passed")


def test_architecture_integrity_validation():
    """Test architecture integrity validation."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    result = validator.validate_architecture_integrity(model)
    
    assert result.status == ValidationStatus.PASSED, f"Architecture integrity validation failed: {result.message}"
    assert result.details['all_components_present'], "All components should be present"
    
    print("PASS: Architecture integrity validation test passed")


def test_parameter_count_validation():
    """Test parameter count validation."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    result = validator.validate_parameter_count(model)
    
    assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING], \
        f"Parameter count validation failed: {result.message}"
    
    print("PASS: Parameter count validation test passed")


def test_config_settings_validation():
    """Test configuration settings validation."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    result = validator.validate_config_settings(model)
    
    assert result.status == ValidationStatus.PASSED, f"Config settings validation failed: {result.message}"
    assert not result.details['issues'], f"Config should have no issues: {result.details['issues']}"
    
    print("PASS: Config settings validation test passed")


def test_capacity_with_optimizations():
    """Test capacity preservation with various optimization settings."""
    base_config = Qwen3VLConfig()
    
    # Test with different optimization configurations
    optimization_configs = [
        # Baseline
        {"name": "baseline", **base_config.__dict__},
        
        # With MoE enabled
        {"name": "moe_enabled", **base_config.__dict__, "use_moe": True, "moe_num_experts": 4, "moe_top_k": 2},
        
        # With Flash Attention 2
        {"name": "flash_attention", **base_config.__dict__, "use_flash_attention_2": True},
        
        # With parameter sharing
        {"name": "param_sharing", **base_config.__dict__, "use_parameter_sharing": True},
        
        # With dynamic sparse attention
        {"name": "sparse_attention", **base_config.__dict__, "use_dynamic_sparse_attention": True},
        
        # With adaptive depth
        {"name": "adaptive_depth", **base_config.__dict__, "use_adaptive_depth": True},
        
        # With KV cache optimization
        {"name": "kv_cache", **base_config.__dict__, "kv_cache_strategy": "hybrid"},
        
        # All optimizations combined
        {
            "name": "all_optimizations", 
            **base_config.__dict__, 
            "use_moe": True, 
            "moe_num_experts": 4, 
            "moe_top_k": 2,
            "use_flash_attention_2": True,
            "use_parameter_sharing": True,
            "use_dynamic_sparse_attention": True,
            "use_adaptive_depth": True,
            "kv_cache_strategy": "hybrid"
        }
    ]
    
    validator = ModelCapacityValidator()
    optimization_validator = OptimizationIntegrationValidator(validator)
    
    for opt_config in optimization_configs:
        # Create config with updated parameters
        config_dict = {k: v for k, v in opt_config.items() if hasattr(Qwen3VLConfig, k)}
        config = Qwen3VLConfig(**config_dict)
        model = create_test_model(config)

        # Validate capacity preservation with this optimization
        result = optimization_validator.validate_optimization_capacity_preservation(
            model,
            opt_config["name"]
        )

        assert result.status == ValidationStatus.PASSED, \
            f"Capacity not preserved with {opt_config['name']}: {result.message}"

        print(f"PASS: Capacity preserved with {opt_config['name']} optimization")

    print("PASS: All optimization capacity preservation tests passed")


def test_capacity_with_invalid_config():
    """Test that invalid configurations are properly detected."""
    # Create a config with wrong layer count
    config_dict = Qwen3VLConfig().__dict__.copy()
    config_dict['num_hidden_layers'] = 16  # Wrong number
    try:
        invalid_config = Qwen3VLConfig(**config_dict)
        # If we get here, the post_init validation didn't catch the error
        model = create_test_model(invalid_config)
        validator = ModelCapacityValidator()
        results = validator.run_comprehensive_validation(model)
        
        # Should have failed layer count validation
        layer_result = next(r for r in results['validation_results'] if r.check_name == "Layer Count Validation")
        assert layer_result.status == ValidationStatus.FAILED, "Should have failed with wrong layer count"
    except ValueError:
        # Expected - the post_init should catch this
        pass
    
    # Create a config with wrong attention head count
    config_dict = Qwen3VLConfig().__dict__.copy()
    config_dict['num_attention_heads'] = 16  # Wrong number
    try:
        invalid_config = Qwen3VLConfig(**config_dict)
        # If we get here, the post_init validation didn't catch the error
        model = create_test_model(invalid_config)
        validator = ModelCapacityValidator()
        results = validator.run_comprehensive_validation(model)
        
        # Should have failed attention heads validation
        heads_result = next(r for r in results['validation_results'] if r.check_name == "Attention Heads Validation")
        assert heads_result.status == ValidationStatus.FAILED, "Should have failed with wrong attention head count"
    except ValueError:
        # Expected - the post_init should catch this
        pass
    
    print("PASS: Invalid configuration detection tests passed")


def test_continuous_monitoring():
    """Test continuous monitoring system."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    monitoring_system = ContinuousMonitoringSystem(validator)
    
    # Monitor at different "training steps"
    contexts = ["initial", "epoch_1", "epoch_5", "final"]
    for context in contexts:
        result = monitoring_system.monitor_model_capacity(model, context)
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING], \
            f"Monitoring failed for context {context}: {result.message}"
    
    # Check drift analysis
    drift_analysis = monitoring_system.check_capacity_drift_over_time()
    assert not drift_analysis['drift_detected'], "No drift should be detected in stable model"
    assert drift_analysis['history_length'] == len(contexts), "Should have recorded all monitoring points"
    
    print("PASS: Continuous monitoring tests passed")


def test_error_reporting():
    """Test error reporting system."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    error_reporter = ErrorReportingSystem()
    
    # Create a validation result that simulates a failure
    failure_result = ValidationResult(
        check_name="Test Capacity Check",
        status=ValidationStatus.FAILED,
        message="Simulated capacity violation for testing",
        details={"test_detail": "test_value"}
    )
    
    # Report the violation
    error_info = error_reporter.report_capacity_violation(failure_result, "test_state")
    
    # Verify error was logged
    assert len(error_reporter.error_log) == 1, "Should have logged one error"
    assert error_info['status'] == 'FAILED', "Error info should have FAILED status"
    assert error_info['severity'] == 'HIGH', "Failed checks should have HIGH severity"
    
    # Generate error report
    error_report = error_reporter.generate_error_report()
    # Check that the report contains the violation information
    assert "Total violations: 1" in error_report, "Report should indicate one violation"
    assert "Test Capacity Check" in error_report, "Report should contain the test check name"

    # Get violation summary
    summary = error_reporter.get_violation_summary()
    assert summary['total_violations'] == 1, "Should have one violation in summary"
    assert summary['failed_checks_count'] == 1, "Should have one failed check"

    print("PASS: Error reporting tests passed")


def test_comprehensive_validation_report():
    """Test comprehensive validation report generation."""
    config = Qwen3VLConfig()
    model = create_test_model(config)
    
    validator = ModelCapacityValidator()
    results = validator.run_comprehensive_validation(model)
    
    # Generate report
    report = validator.generate_validation_report(results)
    
    # Verify report contains expected elements
    assert "Qwen3-VL Model Capacity Validation Report" in report, "Report should have title"
    # Overall status might be WARNING due to parameter count, but capacity should be preserved
    assert "Capacity Preserved: Yes" in report, "Report should indicate capacity preserved"
    assert "Total Checks:" in report, "Report should have summary"
    assert "Overall Status:" in report, "Report should have status"

    print("PASS: Comprehensive validation report test passed")


def test_multiple_optimization_configs():
    """Test validation across multiple optimization configurations."""
    validator = ModelCapacityValidator()
    optimization_validator = OptimizationIntegrationValidator(validator)

    # Define various optimization configurations to test
    optimization_configs = [
        {"name": "baseline", "use_moe": False, "use_flash_attention_2": False},
        {"name": "moe_only", "use_moe": True, "moe_num_experts": 4, "moe_top_k": 2, "use_flash_attention_2": False},
        {"name": "flash_only", "use_moe": False, "use_flash_attention_2": True},
        {"name": "both", "use_moe": True, "moe_num_experts": 4, "moe_top_k": 2, "use_flash_attention_2": True},
    ]

    for config_data in optimization_configs:
        # Create base config and update with optimization settings
        config = Qwen3VLConfig()
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        model = create_test_model(config)

        # Validate this optimization configuration
        results = optimization_validator.validate_multiple_optimizations(model, [config_data])

        # All should pass capacity validation
        for result in results:
            assert result.status == ValidationStatus.PASSED, \
                f"Optimization {config_data['name']} failed capacity validation: {result.message}"

    print("PASS: Multiple optimization configurations test passed")


def run_all_tests():
    """Run all capacity preservation tests."""
    print("Running capacity preservation test suite...")
    print()
    
    test_basic_capacity_preservation()
    test_layer_count_validation()
    test_attention_heads_validation()
    test_architecture_integrity_validation()
    test_parameter_count_validation()
    test_config_settings_validation()
    test_capacity_with_optimizations()
    test_capacity_with_invalid_config()
    test_continuous_monitoring()
    test_error_reporting()
    test_comprehensive_validation_report()
    test_multiple_optimization_configs()
    
    print()
    print("PASS: All capacity preservation tests passed!")


if __name__ == "__main__":
    run_all_tests()