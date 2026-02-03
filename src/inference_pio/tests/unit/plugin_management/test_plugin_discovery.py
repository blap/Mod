"""
Tests for plugin discovery functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.inference_pio.plugins.manager import PluginManager
from tests.unit.plugin_management.test_plugin_helpers import (
    create_mock_plugin_structure,
)


def test_discover_and_load_plugins():
    """Test the automatic discovery and loading of plugins."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock plugin structure
        create_mock_plugin_structure(temp_path)

        # Create plugin manager and discover plugins
        pm = PluginManager()
        count = pm.discover_and_load_plugins(temp_path)

        # Verify that one plugin was loaded
        assert count == 1
        assert "TestModel" in pm.plugins
        assert len(pm.plugins) == 1

        # Verify the plugin is properly registered
        plugin = pm.plugins["TestModel"]
        assert plugin.metadata.name == "TestModel"
        assert plugin.metadata.version == "1.0.0"


def test_discover_and_load_plugins_with_invalid_directory():
    """Test discovery with invalid directory."""
    pm = PluginManager()
    count = pm.discover_and_load_plugins(Path("/invalid/directory"))

    # Should return 0 when directory doesn't exist
    assert count == 0


def test_discover_and_load_plugins_with_no_plugin_files():
    """Test discovery with directory that has no plugin files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a directory without plugin files
        model_dir = temp_path / "empty_model"
        model_dir.mkdir()

        pm = PluginManager()
        count = pm.discover_and_load_plugins(temp_path)

        # Should return 0 when no plugin files exist
        assert count == 0


def test_discover_and_load_plugins_with_multiple_models():
    """Test discovery with multiple model directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple mock model directories
        for i in range(3):
            model_dir = temp_path / f"test_model_{i}"
            model_dir.mkdir(parents=True)

            # Create plugin.py with unique plugin names
            plugin_py_content = f"""
from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface
)
from datetime import datetime
import torch
import torch.nn as nn

class TestModel{i}Plugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="TestModel{i}",
            version="1.0.0",
            author="Test Author",
            description="A test model plugin {i}",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch"],
            compatibility={{"torch_version": ">=2.0.0"}},
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
            return f"Processed: {{data}} (Model {i})"
        elif isinstance(data, (list, tuple)):
            return [f"Processed: {{item}} (Model {i})" for item in data]
        else:
            return f"Processed: {{str(data)}} (Model {i})"

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
        token_map = {{word: idx + 1 for idx, word in enumerate(set(tokens))}}
        return [token_map[word] for word in tokens]

    def detokenize(self, token_ids, **kwargs) -> str:
        if isinstance(token_ids, (list, tuple)):
            return " ".join([f"token_{{tid}}" for tid in token_ids])
        else:
            return f"token_{{token_ids}}"

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        return f"{{prompt}} [GENERATED TEXT Model {i}]"


def create_test_model_{i}_plugin():
    return TestModel{i}Plugin()


__all__ = ["TestModel{i}Plugin", "create_test_model_{i}_plugin"]
"""

            plugin_py_path = model_dir / "plugin.py"
            plugin_py_path.write_text(plugin_py_content)

            # Create plugin_manifest.json
            manifest_content = {
                "name": f"TestModel{i}",
                "version": "1.0.0",
                "author": "Test Author",
                "description": f"A test model plugin {i}",
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
                "tags": ["test", "model"],
                "model_family": "TestFamily",
                "num_parameters": 1000000,
                "test_coverage": 1.0,
                "validation_passed": True,
                "main_class_path": f"src.inference_pio.models.test_model_{i}.plugin.TestModel{i}Plugin",
                "entry_point": f"create_test_model_{i}_plugin",
                "input_types": ["text"],
                "output_types": ["text"],
            }

            manifest_path = model_dir / "plugin_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest_content, f, indent=2)

        # Discover and load plugins
        pm = PluginManager()
        count = pm.discover_and_load_plugins(temp_path)

        # Verify that all three plugins were loaded
        assert count == 3
        assert len(pm.plugins) == 3
        for i in range(3):
            assert f"TestModel{i}" in pm.plugins
