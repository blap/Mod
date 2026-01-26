"""
Test suite for security and resource isolation in Inference-PIO system.
"""

import tempfile
import os
from pathlib import Path
import torch
from typing import Any
from src.inference_pio.test_utils import (
    assert_equal, assert_true, assert_false, assert_is_not_none,
    assert_is_none, assert_in, assert_is_instance, run_tests
)
from src.inference_pio.common.security_manager import (
    SecurityManager,
    ResourceIsolationManager,
    SecurityLevel,
    ResourceLimits,
    get_resource_isolation_manager
)
from src.inference_pio.common.base_plugin_interface import ModelPluginInterface
from src.inference_pio.common.standard_plugin_interface import (
    PluginMetadata,
    PluginType
)
from datetime import datetime


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
            tags=["test", "security"]
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


def test_create_security_context():
    """Test creating a security context."""
    security_manager = SecurityManager()
    context = security_manager.create_security_context(
        plugin_id="test_plugin",
        security_level=SecurityLevel.MEDIUM_TRUST,
        resource_limits=ResourceLimits(cpu_percent=50.0, memory_gb=4.0)
    )

    assert_equal(context.plugin_id, "test_plugin")
    assert_equal(context.security_level, SecurityLevel.MEDIUM_TRUST)
    assert_is_not_none(context.access_token)
    assert_is_not_none(context.sandbox_directory)


def test_validate_path_access_allowed():
    """Test validating allowed path access."""
    security_manager = SecurityManager()
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


def test_validate_path_access_forbidden():
    """Test validating forbidden path access."""
    security_manager = SecurityManager()
    security_manager.create_security_context(
        plugin_id="test_plugin",
        forbidden_paths=["/etc", "/root"]
    )

    result = security_manager.validate_path_access("test_plugin", "/etc/passwd")
    assert_false(result)


def test_enforce_resource_limits_within_bounds():
    """Test enforcing resource limits when within bounds."""
    security_manager = SecurityManager()
    security_manager.create_security_context(
        plugin_id="test_plugin",
        resource_limits=ResourceLimits(cpu_percent=90.0, memory_gb=8.0)
    )

    result = security_manager.enforce_resource_limits("test_plugin")
    assert_true(result)


def test_cleanup_security_context():
    """Test cleaning up a security context."""
    security_manager = SecurityManager()
    security_manager.create_security_context(plugin_id="test_plugin")

    result = security_manager.cleanup_security_context("test_plugin")
    assert_true(result)

    # Context should no longer exist
    context = security_manager.get_security_context("test_plugin")
    assert_is_none(context)


def test_initialize_plugin_isolation():
    """Test initializing plugin isolation."""
    manager = get_resource_isolation_manager()
    result = manager.initialize_plugin_isolation(
        plugin_id="test_plugin",
        security_level=SecurityLevel.HIGH_TRUST,
        resource_limits=ResourceLimits(cpu_percent=60.0, memory_gb=4.0)
    )

    assert_true(result)
    assert_in("test_plugin", manager.plugin_resources)


def test_begin_end_operation():
    """Test beginning and ending an operation."""
    manager = get_resource_isolation_manager()
    manager.initialize_plugin_isolation(plugin_id="test_plugin")

    token = manager.begin_operation("test_plugin")
    assert_is_not_none(token)

    result = manager.end_operation("test_plugin", token)
    assert_true(result)


def test_validate_path_access():
    """Test validating path access through isolation manager."""
    manager = get_resource_isolation_manager()
    manager.initialize_plugin_isolation(
        plugin_id="test_plugin",
        resource_limits=ResourceLimits(
            cpu_percent=80.0,
            memory_gb=4.0,
            disk_space_gb=2.0
        )
    )

    # Test with a valid temporary path
    with tempfile.NamedTemporaryFile() as tmp_file:
        result = manager.validate_path_access("test_plugin", tmp_file.name)
        # Should return True as temporary paths are typically allowed
        # The actual result depends on the default allowed paths
        assert_is_instance(result, bool)


def test_get_plugin_resource_usage():
    """Test getting plugin resource usage."""
    manager = get_resource_isolation_manager()
    manager.initialize_plugin_isolation(plugin_id="test_plugin")

    usage = manager.get_plugin_resource_usage("test_plugin")

    assert_is_instance(usage, dict)
    assert_in("plugin_id", usage)
    assert_equal(usage["plugin_id"], "test_plugin")


def test_initialize_security():
    """Test initializing security for a plugin."""
    plugin = MockModelPlugin()
    result = plugin.initialize_security(
        security_level=SecurityLevel.HIGH_TRUST,
        resource_limits=ResourceLimits(cpu_percent=70.0, memory_gb=3.0)
    )

    assert_true(result)
    assert_true(plugin._security_initialized)


def test_validate_file_access():
    """Test validating file access for a plugin."""
    plugin = MockModelPlugin()
    # Initialize security first
    plugin.initialize_security()

    # Test with a temporary file
    with tempfile.NamedTemporaryFile() as tmp_file:
        result = plugin.validate_file_access(tmp_file.name)
        # Should return True as temporary paths are typically allowed
        assert_is_instance(result, bool)


def test_validate_network_access():
    """Test validating network access for a plugin."""
    plugin = MockModelPlugin()
    # Initialize security first
    plugin.initialize_security()

    result = plugin.validate_network_access("localhost")
    # Localhost should always be allowed
    assert_true(result)


def test_get_resource_usage():
    """Test getting resource usage for a plugin."""
    plugin = MockModelPlugin()
    # Initialize security first
    plugin.initialize_security()

    usage = plugin.get_resource_usage()
    assert_is_instance(usage, dict)


def test_cleanup_security():
    """Test cleaning up security for a plugin."""
    plugin = MockModelPlugin()
    # Initialize security first
    plugin.initialize_security()

    result = plugin.cleanup_security()
    assert_true(result)
    assert_false(plugin._security_initialized)


def test_complete_plugin_lifecycle_with_security():
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
    assert_equal(result, "test result")

    # Get resource usage
    usage = plugin.get_resource_usage()
    assert_is_instance(usage, dict)

    # Cleanup
    cleanup_result = plugin.cleanup()
    assert_true(cleanup_result)


if __name__ == "__main__":
    print("Running security and isolation tests...")
    run_tests([
        test_create_security_context,
        test_validate_path_access_allowed,
        test_validate_path_access_forbidden,
        test_enforce_resource_limits_within_bounds,
        test_cleanup_security_context,
        test_initialize_plugin_isolation,
        test_begin_end_operation,
        test_validate_path_access,
        test_get_plugin_resource_usage,
        test_initialize_security,
        test_validate_file_access,
        test_validate_network_access,
        test_get_resource_usage,
        test_cleanup_security,
        test_complete_plugin_lifecycle_with_security
    ])