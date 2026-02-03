"""
Tests for the dynamic loading system in src/inference_pio/plugins/manager.py
"""

import importlib.util
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface,
)
from src.inference_pio.plugins.manager import PluginManager


def create_test_plugin_structure(base_path: Path):
    """Create a test plugin structure for dynamic loading tests."""

    # Create a model directory
    model_dir = base_path / "test_dynamic_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a plugin module with a test plugin class
    plugin_content = """
from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface
)
from datetime import datetime
import torch
import torch.nn as nn

class TestDynamicPlugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="TestDynamicPlugin",
            version="1.0.0",
            author="Test Author",
            description="A test model plugin for dynamic loading",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch"],
            compatibility={"torch_version": ">=2.0.0"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        super().__init__(metadata)
        self._initialized = False
        self._model = None

    def initialize(self, **kwargs) -> bool:
        self._initialized = True
        return True

    def load_model(self, config=None):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        self._model = SimpleModel()
        return self._model

    def infer(self, data):
        if not self._initialized:
            self.initialize()

        if isinstance(data, str):
            return f"DYNAMIC PROCESSED: {data}"
        elif isinstance(data, (list, tuple)):
            return [f"DYNAMIC PROCESSED: {item}" for item in data]
        else:
            return f"DYNAMIC PROCESSED: {str(data)}"

    def cleanup(self) -> bool:
        self._model = None
        self._initialized = False
        return True

    def supports_config(self, config) -> bool:
        return config is None or isinstance(config, dict)

    def tokenize(self, text: str, **kwargs):
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        tokens = text.split()
        token_map = {word: idx + 1 for idx, word in enumerate(set(tokens))}
        return [token_map[word] for word in tokens]

    def detokenize(self, token_ids, **kwargs) -> str:
        if isinstance(token_ids, (list, tuple)):
            return " ".join([f"token_{tid}" for tid in token_ids])
        else:
            return f"token_{token_ids}"

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        return f"{prompt} [DYNAMIC LOADING RESULT]"


def create_test_dynamic_plugin():
    return TestDynamicPlugin()


__all__ = ["TestDynamicPlugin", "create_test_dynamic_plugin"]
"""

    plugin_path = model_dir / "dynamic_plugin.py"
    plugin_path.write_text(plugin_content)

    # Create plugin manifest
    manifest_content = {
        "name": "TestDynamicPlugin",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "A test model plugin for dynamic loading",
        "plugin_type": "MODEL_COMPONENT",
        "dependencies": ["torch"],
        "compatibility": {"torch_version": ">=2.0.0"},
        "created_at": "2026-01-31T00:00:00",
        "updated_at": "2026-01-31T00:00:00",
        "model_architecture": "TestArch",
        "model_size": "1.0B",
        "required_memory_gb": 1.0,
        "supported_modalities": ["text"],
        "license": "MIT",
        "tags": ["test", "model", "dynamic"],
        "model_family": "TestFamily",
        "num_parameters": 1000000,
        "test_coverage": 1.0,
        "validation_passed": True,
        "main_class_path": f"src.inference_pio.models.test_dynamic_model.dynamic_plugin.TestDynamicPlugin",
        "entry_point": "create_test_dynamic_plugin",
        "input_types": ["text"],
        "output_types": ["text"],
    }

    manifest_path = model_dir / "plugin_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_content, f, indent=2)

    # Create __init__.py
    init_path = model_dir / "__init__.py"
    init_path.write_text("# Test dynamic model init file")


def test_dynamic_class_loading_by_main_class_path():
    """Test loading a plugin by its main class path from manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test plugin structure
        create_test_plugin_structure(temp_path)

        # Modify the manifest to use a temporary path
        model_dir = temp_path / "test_dynamic_model"
        manifest_path = model_dir / "plugin_manifest.json"

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Update the main_class_path to reference the temp location
        # We'll temporarily add the temp path to sys.path for this test
        import sys

        original_path = sys.path[:]
        try:
            # Add temp directory to Python path so we can import from it
            sys.path.insert(0, str(temp_path))

            # Update manifest to use temp path
            manifest["main_class_path"] = (
                f"test_dynamic_model.dynamic_plugin.TestDynamicPlugin"
            )

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Create plugin manager and test dynamic loading
            pm = PluginManager()

            # Test the internal method for loading by class path
            plugin_instance = pm._load_plugin_by_class_path(
                model_dir, "test_dynamic_model.dynamic_plugin.TestDynamicPlugin"
            )

            # Verify the plugin was loaded correctly
            assert plugin_instance is not None
            assert plugin_instance.metadata.name == "TestDynamicPlugin"
            assert isinstance(plugin_instance, TextModelPluginInterface)

        finally:
            # Restore original path
            sys.path[:] = original_path


def test_dynamic_entry_point_loading():
    """Test loading a plugin by its entry point function from manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test plugin structure
        create_test_plugin_structure(temp_path)

        # Modify the manifest to use a temporary path
        model_dir = temp_path / "test_dynamic_model"
        manifest_path = model_dir / "plugin_manifest.json"

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Update the main_class_path to reference the temp location
        import sys

        original_path = sys.path[:]
        try:
            # Add temp directory to Python path so we can import from it
            sys.path.insert(0, str(temp_path))

            # Update manifest to use temp path
            manifest["main_class_path"] = (
                f"test_dynamic_model.dynamic_plugin.TestDynamicPlugin"
            )

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Create plugin manager and test dynamic loading
            pm = PluginManager()

            # Test the internal method for loading by entry point
            plugin_instance = pm._load_plugin_by_entry_point(
                model_dir,
                "test_dynamic_model.dynamic_plugin",
                "create_test_dynamic_plugin",
            )

            # Verify the plugin was loaded correctly
            assert plugin_instance is not None
            assert plugin_instance.metadata.name == "TestDynamicPlugin"
            assert isinstance(plugin_instance, TextModelPluginInterface)

        finally:
            # Restore original path
            sys.path[:] = original_path


def test_load_plugin_from_manifest():
    """Test the complete manifest-based plugin loading process."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test plugin structure
        create_test_plugin_structure(temp_path)

        # Modify the manifest to use a temporary path
        model_dir = temp_path / "test_dynamic_model"
        manifest_path = model_dir / "plugin_manifest.json"

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Update the main_class_path to reference the temp location
        import sys

        original_path = sys.path[:]
        try:
            # Add temp directory to Python path so we can import from it
            sys.path.insert(0, str(temp_path))

            # Update manifest to use temp path
            manifest["main_class_path"] = (
                f"test_dynamic_model.dynamic_plugin.TestDynamicPlugin"
            )

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Create plugin manager and test manifest loading
            pm = PluginManager()

            # Call the internal method to load from manifest
            loaded_count = pm._load_plugin_from_manifest(model_dir, manifest_path)

            # Verify the plugin was loaded
            assert loaded_count == 1
            assert "TestDynamicPlugin" in pm.plugins
            plugin = pm.plugins["TestDynamicPlugin"]
            assert plugin.metadata.name == "TestDynamicPlugin"
            assert isinstance(plugin, TextModelPluginInterface)

        finally:
            # Restore original path
            sys.path[:] = original_path


def test_discover_and_load_plugins_with_dynamic_loading():
    """Test the full discovery and dynamic loading process."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test plugin structure
        create_test_plugin_structure(temp_path)

        import sys

        original_path = sys.path[:]
        try:
            # Add temp directory to Python path so we can import from it
            sys.path.insert(0, str(temp_path))

            # Update the manifest to use the correct path
            model_dir = temp_path / "test_dynamic_model"
            manifest_path = model_dir / "plugin_manifest.json"

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["main_class_path"] = (
                f"test_dynamic_model.dynamic_plugin.TestDynamicPlugin"
            )

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Test the full discovery and loading process
            pm = PluginManager()
            count = pm.discover_and_load_plugins(temp_path)

            # Verify the plugin was discovered and loaded
            assert count == 1
            assert len(pm.plugins) == 1
            assert "TestDynamicPlugin" in pm.plugins

            plugin = pm.plugins["TestDynamicPlugin"]
            assert plugin.metadata.name == "TestDynamicPlugin"
            assert isinstance(plugin, TextModelPluginInterface)

            # Test that the plugin can be activated
            activation_result = pm.activate_plugin("TestDynamicPlugin")
            assert activation_result is True
            assert "TestDynamicPlugin" in pm.active_plugins

        finally:
            # Restore original path
            sys.path[:] = original_path


def test_dynamic_loading_with_nonexistent_class():
    """Test handling of nonexistent classes in manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a model directory with a manifest that references a nonexistent class
        model_dir = temp_path / "nonexistent_model"
        model_dir.mkdir()

        # Create plugin manifest with nonexistent class
        manifest_content = {
            "name": "NonExistentPlugin",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "A test model plugin with nonexistent class",
            "plugin_type": "MODEL_COMPONENT",
            "dependencies": ["torch"],
            "compatibility": {"torch_version": ">=2.0.0"},
            "created_at": "2026-01-31T00:00:00",
            "updated_at": "2026-01-31T00:00:00",
            "main_class_path": "nonexistent.module.NonExistentClass",
            "entry_point": "nonexistent_function",
            "input_types": ["text"],
            "output_types": ["text"],
        }

        manifest_path = model_dir / "plugin_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_content, f, indent=2)

        # Test that the system handles nonexistent classes gracefully
        pm = PluginManager()
        loaded_count = pm._load_plugin_from_manifest(model_dir, manifest_path)

        # Should return 0 since the class doesn't exist
        assert loaded_count == 0


def test_dynamic_loading_with_invalid_manifest():
    """Test handling of invalid manifest files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a model directory with an invalid manifest
        model_dir = temp_path / "invalid_model"
        model_dir.mkdir()

        # Create an invalid manifest file
        manifest_path = model_dir / "plugin_manifest.json"
        with open(manifest_path, "w") as f:
            f.write("{ invalid json")

        # Test that the system handles invalid manifests gracefully
        pm = PluginManager()
        loaded_count = pm._load_plugin_from_manifest(model_dir, manifest_path)

        # Should return 0 due to invalid JSON
        assert loaded_count == 0


if __name__ == "__main__":
    pytest.main([__file__])
