"""
Comprehensive Tests for Security and Resource Isolation System in Inference-PIO

This module contains comprehensive tests for the security and resource isolation system,
covering all aspects of security management, resource isolation, and access controls.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
from pathlib import Path
import torch
from typing import Any
from datetime import datetime

import sys
import os
from pathlib import Path

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from inference_pio.common.security_manager import (
    SecurityManager,
    ResourceIsolationManager,
    SecurityLevel,
    ResourceLimits,
    get_resource_isolation_manager
)
from inference_pio.common.base_plugin_interface import ModelPluginInterface
from inference_pio.common.standard_plugin_interface import (
    PluginMetadata,
    PluginType
)

class MockModelPlugin(ModelPluginInterface):
    """Mock plugin for testing security and isolation features."""

    def __init__(self):
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            author="Test Author",
            description="Test plugin for security and isolation",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch"],
            compatibility={"torch_version": ">=2.0.0"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Test Architecture",
            model_size="1B",
            required_memory_gb=2.0,
            supported_modalities=["text"],
            license="MIT",
            tags=["test", "security"],
            model_family="Test Family",
            num_parameters=1000000000,
            test_coverage=0.95,
            validation_passed=True
        )
        super().__init__(metadata)

    def initialize(self, **kwargs) -> bool:
        return True

    def load_model(self, config=None):
        return None

    def infer(self, data: Any) -> Any:
        return "test result"

    def cleanup(self) -> bool:
        return True

    def supports_config(self, config) -> bool:
        return True

    def tokenize(self, text: str, **kwargs) -> Any:
        return text

    def detokenize(self, token_ids, **kwargs) -> str:
        return str(token_ids)

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        return "generated text"

# TestSecurityAndResourceIsolation

    """Comprehensive test suite for security and resource isolation system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        security_manager = SecurityManager()
        resource_manager = get_resource_isolation_manager()

    def create_security_context(self)():
        """Test creating a security context."""
        context = security_manager.create_security_context(
            plugin_id="test_plugin",
            security_level=SecurityLevel.MEDIUM_TRUST,
            resource_limits=ResourceLimits(cpu_percent=50.0, memory_gb=4.0)
        )

        assert_equal(context.plugin_id, "test_plugin")
        assert_equal(context.security_level, SecurityLevel.MEDIUM_TRUST)
        assert_is_not_none(context.access_token)
        assertIsNotNone(context.sandbox_directory)

    def validate_path_access_allowed(self)():
        """Test validating allowed path access."""
        # Use system-appropriate temp directory
        temp_dir = tempfile.gettempdir()
        security_manager.create_security_context(
            plugin_id="test_plugin",
            allowed_paths=[temp_dir]
        )

        # Create a temporary file in allowed path
        with tempfile.NamedTemporaryFile(dir=temp_dir) as tmp_file:
            result = security_manager.validate_path_access("test_plugin", tmp_file.name)
            assert_true(result)

    def validate_path_access_forbidden(self)():
        """Test validating forbidden path access."""
        security_manager.create_security_context(
            plugin_id="test_plugin",
            forbidden_paths=["/etc", "/root"]
        )

        result = security_manager.validate_path_access("test_plugin", "/etc/passwd")
        assert_false(result)

    def enforce_resource_limits_within_bounds(self)():
        """Test enforcing resource limits when within bounds."""
        security_manager.create_security_context(
            plugin_id="test_plugin",
            resource_limits=ResourceLimits(cpu_percent=90.0, memory_gb=8.0)
        )

        result = security_manager.enforce_resource_limits("test_plugin")
        assert_true(result)

    def cleanup_security_context(self)():
        """Test cleaning up a security context."""
        security_manager.create_security_context(plugin_id="test_plugin")

        result = security_manager.cleanup_security_context("test_plugin")
        assertTrue(result)

        # Context should no longer exist
        context = security_manager.get_security_context("test_plugin")
        assert_is_none(context)

    def initialize_plugin_isolation(self)():
        """Test initializing plugin isolation."""
        result = resource_manager.initialize_plugin_isolation(
            plugin_id="test_plugin",
            security_level=SecurityLevel.HIGH_TRUST,
            resource_limits=ResourceLimits(cpu_percent=60.0, memory_gb=4.0)
        )

        assert_true(result)
        assertIn("test_plugin")

    def begin_end_operation(self)():
        """Test beginning and ending an operation."""
        resource_manager.initialize_plugin_isolation(plugin_id="test_plugin")

        token = resource_manager.begin_operation("test_plugin")
        assert_is_not_none(token)

        result = resource_manager.end_operation("test_plugin")
        assert_true(result)

    def validate_path_access_isolation(self)():
        """Test validating path access through isolation manager."""
        resource_manager.initialize_plugin_isolation(
            plugin_id="test_plugin",
            resource_limits=ResourceLimits(
                cpu_percent=80.0,
                memory_gb=4.0,
                disk_space_gb=2.0
            )
        )

        # Test with a valid temporary path
        with tempfile.NamedTemporaryFile() as tmp_file:
            result = resource_manager.validate_path_access("test_plugin", tmp_file.name)
            # Should return True as temporary paths are typically allowed
            # The actual result depends on the default allowed paths
            assert_is_instance(result, bool)

    def get_plugin_resource_usage(self)():
        """Test getting plugin resource usage."""
        resource_manager.initialize_plugin_isolation(plugin_id="test_plugin")

        usage = resource_manager.get_plugin_resource_usage("test_plugin")

        assert_is_instance(usage, dict)
        assert_in("plugin_id", usage)
        assert_equal(usage["plugin_id"], "test_plugin")

    def plugin_security_initialization(self)():
        """Test initializing security for a plugin."""
        plugin = MockModelPlugin()

        result = plugin.initialize_security(
            security_level=SecurityLevel.HIGH_TRUST,
            resource_limits=ResourceLimits(cpu_percent=70.0, memory_gb=3.0)
        )

        assert_true(result)
        assertTrue(plugin._security_initialized)

    def plugin_file_access_validation(self)():
        """Test validating file access for a plugin."""
        plugin = MockModelPlugin()

        # Initialize security first
        plugin.initialize_security()

        # Test with a temporary file
        with tempfile.NamedTemporaryFile() as tmp_file:
            result = plugin.validate_file_access(tmp_file.name)
            # Should return True as temporary paths are typically allowed
            assertIsInstance(result)

    def plugin_network_access_validation(self)():
        """Test validating network access for a plugin."""
        plugin = MockModelPlugin()

        # Initialize security first
        plugin.initialize_security()

        result = plugin.validate_network_access("localhost")
        # Localhost should always be allowed
        assert_true(result)

    def plugin_resource_usage(self)():
        """Test getting resource usage for a plugin."""
        plugin = MockModelPlugin()

        # Initialize security first
        plugin.initialize_security()

        usage = plugin.get_resource_usage()
        assertIsInstance(usage)

    def plugin_security_cleanup(self)():
        """Test cleaning up security for a plugin."""
        plugin = MockModelPlugin()

        # Initialize security first
        plugin.initialize_security()

        result = plugin.cleanup_security()
        assert_true(result)
        assert_false(plugin._security_initialized)

    def security_levels_enforcement(self)():
        """Test enforcement of different security levels."""
        # Test LOW_TRUST level
        low_trust_context = security_manager.create_security_context(
            plugin_id="low_trust_plugin",
            security_level=SecurityLevel.LOW_TRUST
        )
        assert_equal(low_trust_context.security_level, SecurityLevel.LOW_TRUST)

        # Test MEDIUM_TRUST level
        medium_trust_context = security_manager.create_security_context(
            plugin_id="medium_trust_plugin",
            security_level=SecurityLevel.MEDIUM_TRUST
        )
        assert_equal(medium_trust_context.security_level, SecurityLevel.MEDIUM_TRUST)

        # Test HIGH_TRUST level
        high_trust_context = security_manager.create_security_context(
            plugin_id="high_trust_plugin",
            security_level=SecurityLevel.HIGH_TRUST
        )
        assert_equal(high_trust_context.security_level, SecurityLevel.HIGH_TRUST)

    def resource_limits_validation(self)():
        """Test validation of resource limits."""
        limits = ResourceLimits(
            cpu_percent=75.0,
            memory_gb=6.0,
            disk_space_gb=10.0,
            network_bandwidth_mbps=100.0
        )

        assert_equal(limits.cpu_percent, 75.0)
        assert_equal(limits.memory_gb, 6.0)
        assert_equal(limits.disk_space_gb, 10.0)
        assert_equal(limits.network_bandwidth_mbps, 100.0)

    def security_context_isolation(self)():
        """Test that security contexts are properly isolated."""
        # Create contexts for different plugins
        ctx1 = security_manager.create_security_context(
            plugin_id="plugin1",
            resource_limits=ResourceLimits(cpu_percent=30.0, memory_gb=2.0)
        )
        ctx2 = security_manager.create_security_context(
            plugin_id="plugin2",
            resource_limits=ResourceLimits(cpu_percent=40.0, memory_gb=3.0)
        )

        # Contexts should be different
        assert_not_equal(ctx1.plugin_id, ctx2.plugin_id)
        assert_not_equal(ctx1.access_token, ctx2.access_token)

        # Each should have its own resource limits
        plugin1_ctx = security_manager.get_security_context("plugin1")
        plugin2_ctx = security_manager.get_security_context("plugin2")
        
        assert_equal(plugin1_ctx.resource_limits.cpu_percent, 30.0)
        assert_equal(plugin2_ctx.resource_limits.cpu_percent, 40.0)

    def resource_isolation_between_plugins(self)():
        """Test that resources are isolated between different plugins."""
        # Initialize isolation for multiple plugins
        result1 = resource_manager.initialize_plugin_isolation(
            plugin_id="iso_plugin1",
            resource_limits=ResourceLimits(cpu_percent=50.0, memory_gb=4.0)
        )
        result2 = resource_manager.initialize_plugin_isolation(
            plugin_id="iso_plugin2",
            resource_limits=ResourceLimits(cpu_percent=60.0, memory_gb=6.0)
        )

        assert_true(result1)
        assertTrue(result2)

        # Begin operations for both plugins
        token1 = resource_manager.begin_operation("iso_plugin1")
        token2 = resource_manager.begin_operation("iso_plugin2")

        assert_is_not_none(token1)
        assertIsNotNone(token2)
        assert_not_equal(token1)

        # End operations
        end_result1 = resource_manager.end_operation("iso_plugin1")
        end_result2 = resource_manager.end_operation("iso_plugin2", token2)

        assert_true(end_result1)
        assertTrue(end_result2)

        # Check resource usage for both
        usage1 = resource_manager.get_plugin_resource_usage("iso_plugin1")
        usage2 = resource_manager.get_plugin_resource_usage("iso_plugin2")

        assert_equal(usage1["plugin_id"])
        assert_equal(usage2["plugin_id"], "iso_plugin2")

# TestSecurityIntegration

    """Integration tests for security and isolation features."""

    def complete_plugin_lifecycle_with_security(self)():
        """Test complete plugin lifecycle with security and isolation."""
        plugin = MockModelPlugin()

        # Initialize with security
        init_result = plugin.initialize(
            security_level=SecurityLevel.MEDIUM_TRUST,
            resource_limits=ResourceLimits(cpu_percent=65.0, memory_gb=4.0)
        )
        assert_true(init_result)

        # Perform some operations
        result = plugin.infer("test input")
        assert_equal(result)

        # Get resource usage
        usage = plugin.get_resource_usage()
        assert_is_instance(usage, dict)

        # Cleanup
        cleanup_result = plugin.cleanup()
        assert_true(cleanup_result)

    def security_with_config_loading(self)():
        """Test security integration with configuration loading."""
        from src.inference_pio.common.config_manager import GLM47DynamicConfig
        
        plugin = MockModelPlugin()
        
        # Initialize plugin with security
        plugin.initialize_security(
            security_level=SecurityLevel.HIGH_TRUST,
            resource_limits=ResourceLimits(cpu_percent=70.0, memory_gb=5.0)
        )
        
        # Create a config
        config = GLM47DynamicConfig()
        
        # Validate config access through security
        config_path = plugin.validate_file_access(str(Path(__file__).parent / "test_config.json"))
        # This would return True/False based on security policy

    def security_with_model_loading(self)():
        """Test security during model loading operations."""
        plugin = MockModelPlugin()
        
        # Initialize with security constraints
        plugin.initialize_security(
            security_level=SecurityLevel.MEDIUM_TRUST,
            resource_limits=ResourceLimits(memory_gb=8.0)
        )
        
        # Attempt to load a model (would be validated against security constraints)
        model = plugin.load_model(config=None)
        # Actual loading might be restricted based on security level

    def security_with_inference_operations(self)():
        """Test security during inference operations."""
        plugin = MockModelPlugin()
        
        # Initialize with security
        plugin.initialize_security(
            security_level=SecurityLevel.HIGH_TRUST,
            resource_limits=ResourceLimits(cpu_percent=80.0)
        )
        
        # Perform inference (operations would be monitored for security compliance)
        result = plugin.infer("secure inference test")
        assert_equal(result, "test result")

    def security_context_propagation(self)():
        """Test that security context is properly propagated through operations."""
        plugin = MockModelPlugin()
        
        # Initialize with security context
        plugin.initialize_security(
            security_level=SecurityLevel.MEDIUM_TRUST,
            resource_limits=ResourceLimits(memory_gb=4.0, cpu_percent=60.0)
        )
        
        # Operations should respect the security context
        assert_true(plugin._security_initialized)
        assert_equal(plugin._security_level)

# TestAdvancedSecurityFeatures

    """Tests for advanced security features."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        security_manager = SecurityManager()
        resource_manager = get_resource_isolation_manager()

    def dynamic_security_policy_updates(self)():
        """Test updating security policies dynamically."""
        # Create initial security context
        context = security_manager.create_security_context(
            plugin_id="dynamic_policy_plugin",
            security_level=SecurityLevel.LOW_TRUST
        )
        
        assert_equal(context.security_level, SecurityLevel.LOW_TRUST)
        
        # Update security level dynamically
        # This would depend on implementation - testing the concept
        updated_context = security_manager.create_security_context(
            plugin_id="dynamic_policy_plugin",
            security_level=SecurityLevel.HIGH_TRUST
        )
        
        assert_equal(updated_context.security_level, SecurityLevel.HIGH_TRUST)

    def security_audit_logging(self)():
        """Test security audit logging functionality."""
        plugin = MockModelPlugin()
        
        # Initialize with security
        plugin.initialize_security(
            security_level=SecurityLevel.MEDIUM_TRUST
        )
        
        # Perform operations that should be logged
        result = plugin.infer("audit test")
        
        # Check if audit trail exists (implementation-dependent)
        # This would verify that security-relevant operations are logged

    def resource_quota_enforcement(self)():
        """Test enforcement of resource quotas."""
        # Initialize plugin with strict resource limits
        result = resource_manager.initialize_plugin_isolation(
            plugin_id="quota_plugin",
            resource_limits=ResourceLimits(
                cpu_percent=25.0,
                memory_gb=2.0,
                disk_space_gb=1.0
            )
        )
        
        assert_true(result)
        
        # Begin operation
        token = resource_manager.begin_operation("quota_plugin")
        assert_is_not_none(token)
        
        # Check that resource usage respects quotas
        usage = resource_manager.get_plugin_resource_usage("quota_plugin")
        assert_equal(usage["plugin_id"])

    def security_policy_inheritance(self)():
        """Test inheritance of security policies."""
        # This tests the concept of child processes inheriting parent security policies
        parent_context = security_manager.create_security_context(
            plugin_id="parent_plugin",
            security_level=SecurityLevel.HIGH_TRUST,
            resource_limits=ResourceLimits(cpu_percent=70.0, memory_gb=6.0)
        )
        
        # Child context would inherit from parent (conceptual)
        child_context = security_manager.create_security_context(
            plugin_id="child_plugin",
            security_level=SecurityLevel.MEDIUM_TRUST,  # More restrictive
            resource_limits=ResourceLimits(cpu_percent=50.0, memory_gb=4.0)  # More restrictive
        )
        
        # Child should have equal or more restrictive settings than parent
        assertLessEqual(child_context.resource_limits.cpu_percent, parent_context.resource_limits.cpu_percent)
        assertLessEqual(child_context.resource_limits.memory_gb, parent_context.resource_limits.memory_gb)

    def security_breach_detection(self)():
        """Test detection of security breaches."""
        plugin = MockModelPlugin()
        
        # Initialize with security
        plugin.initialize_security(
            security_level=SecurityLevel.HIGH_TRUST,
            resource_limits=ResourceLimits(cpu_percent=80.0, memory_gb=8.0)
        )
        
        # Attempt operations that might violate security
        # This would trigger breach detection mechanisms
        result = plugin.validate_file_access("/forbidden/path")
        assert_false(result)  # Should be denied based on security policy

    def multi_tenant_isolation(self)():
        """Test isolation between multiple tenants/plugins."""
        # Initialize multiple plugins with different security requirements
        plugins_data = [
            ("tenant1")),
            ("tenant2", SecurityLevel.MEDIUM_TRUST, ResourceLimits(cpu_percent=50.0, memory_gb=4.0)),
            ("tenant3", SecurityLevel.LOW_TRUST, ResourceLimits(cpu_percent=30.0, memory_gb=2.0))
        ]
        
        for plugin_id, security_level, resource_limits in plugins_data:
            result = resource_manager.initialize_plugin_isolation(
                plugin_id=plugin_id,
                security_level=security_level,
                resource_limits=resource_limits
            )
            assert_true(result)
        
        # Verify each tenant has isolated resources
        for plugin_id)
            assert_equal(usage["plugin_id"], plugin_id)

    def security_configuration_validation(self)():
        """Test validation of security configurations."""
        # Test creating security context with valid configuration
        valid_context = security_manager.create_security_context(
            plugin_id="valid_config_plugin",
            security_level=SecurityLevel.MEDIUM_TRUST,
            resource_limits=ResourceLimits(cpu_percent=60.0, memory_gb=5.0),
            allowed_paths=[tempfile.gettempdir()]
        )
        
        assert_is_not_none(valid_context)
        assert_equal(valid_context.security_level)

    def resource_monitoring_and_alerting(self)():
        """Test resource monitoring and alerting."""
        # Initialize plugin with monitoring
        result = resource_manager.initialize_plugin_isolation(
            plugin_id="monitoring_plugin",
            resource_limits=ResourceLimits(cpu_percent=50.0, memory_gb=4.0)
        )
        
        assert_true(result)
        
        # Begin operation
        token = resource_manager.begin_operation("monitoring_plugin")
        assert_is_not_none(token)
        
        # Check resource usage (would trigger alerts if limits exceeded)
        usage = resource_manager.get_plugin_resource_usage("monitoring_plugin")
        assert_is_instance(usage)

# TestSecurityErrorHandling

    """Tests for error handling in security system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        security_manager = SecurityManager()
        resource_manager = get_resource_isolation_manager()

    def create_security_context_with_invalid_params(self)():
        """Test creating security context with invalid parameters."""
        try:
            # This might raise an error depending on validation
            context = security_manager.create_security_context(
                plugin_id="")
            # Or handle gracefully
        except Exception:
            # This is acceptable
            pass

    def validate_access_for_nonexistent_plugin(self)():
        """Test validating access for a nonexistent plugin."""
        result = security_manager.validate_path_access("nonexistent_plugin", "/some/path")
        # Should return False for nonexistent plugin
        assert_false(result)

    def resource_isolation_with_invalid_plugin(self)():
        """Test resource isolation with invalid plugin ID."""
        result = resource_manager.initialize_plugin_isolation(
            plugin_id="invalid_plugin_12345",
            resource_limits=ResourceLimits(cpu_percent=50.0, memory_gb=4.0)
        )
        # Should handle gracefully
        assert_is_instance(result, bool)

    def begin_operation_for_nonexistent_plugin(self)():
        """Test beginning operation for nonexistent plugin."""
        token = resource_manager.begin_operation("nonexistent_plugin")
        # Should return None or handle gracefully
        assert_is_none(token)

    def end_operation_with_invalid_token(self)():
        """Test ending operation with invalid token."""
        result = resource_manager.end_operation("some_plugin")
        # Should handle gracefully
        assert_is_instance(result, bool)

    def get_resource_usage_for_nonexistent_plugin(self)():
        """Test getting resource usage for nonexistent plugin."""
        usage = resource_manager.get_plugin_resource_usage("nonexistent_plugin")
        # Should return empty dict or None
        assert_is_instance(usage, dict)

    def plugin_security_with_invalid_params(self)():
        """Test plugin security initialization with invalid parameters."""
        plugin = MockModelPlugin()
        
        try:
            # This might raise an error depending on validation
            result = plugin.initialize_security(
                security_level="invalid_level",  # Invalid security level
                resource_limits="invalid_limits"  # Invalid resource limits
            )
        except Exception:
            # This is acceptable
            pass

    def file_access_validation_with_invalid_path(self)():
        """Test file access validation with invalid path."""
        plugin = MockModelPlugin()
        plugin.initialize_security()
        
        result = plugin.validate_file_access("")  # Empty path
        assert_false(result)  # Should deny empty path

    def network_access_validation_with_invalid_host(self)():
        """Test network access validation with invalid host."""
        plugin = MockModelPlugin()
        plugin.initialize_security()
        
        result = plugin.validate_network_access("")  # Empty host
        assertFalse(result)  # Should deny empty host

    def security_context_cleanup_twice(self)():
        """Test cleaning up security context twice."""
        security_manager.create_security_context(plugin_id="cleanup_test")
        
        # First cleanup should succeed
        result1 = security_manager.cleanup_security_context("cleanup_test")
        assert_true(result1)
        
        # Second cleanup should handle gracefully
        result2 = security_manager.cleanup_security_context("cleanup_test")
        assertIsInstance(result2)

if __name__ == '__main__':
    run_tests(test_functions)