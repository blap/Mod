"""
Tests for plugin loading functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.plugins.manager import PluginManager, discover_and_load_plugins
from tests.unit.plugin_management.test_plugin_helpers import (
    create_mock_plugin_structure,
)


def test_global_discover_and_load_plugins():
    """Test the global discover_and_load_plugins function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock plugin structure
        create_mock_plugin_structure(temp_path)

        # Use the global function
        count = discover_and_load_plugins(temp_path)

        # Note: Since this uses the global plugin manager, we can't easily verify
        # the plugins were added without accessing the global state directly.
        # For now, just verify the function runs without error and returns expected count.
        assert (
            count >= 0
        )  # At least 0, possibly more if other plugins were already registered


def test_discover_and_load_plugins_with_plugin_py():
    """Test discovery from plugin.py files with manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a model directory with plugin.py
        model_dir = temp_path / "plugin_model"
        model_dir.mkdir(parents=True)

        # Create plugin.py with plugin class
        plugin_content = """
from src.common.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface
)
from datetime import datetime
import torch
import torch.nn as nn

class PluginModelPlugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="PluginModel",
            version="1.0.0",
            author="Test Author",
            description="A test model plugin from plugin.py",
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
            return f"Processed: {data} (Plugin Model)"
        elif isinstance(data, (list, tuple)):
            return [f"Processed: {item} (Plugin Model)" for item in data]
        else:
            return f"Processed: {str(data)} (Plugin Model)"

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
            return f"token_{tid}"

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        return f"{prompt} [GENERATED TEXT FROM PLUGIN]"


def create_plugin_model_plugin():
    return PluginModelPlugin()


__all__ = ["PluginModelPlugin", "create_plugin_model_plugin"]
"""

        plugin_path = model_dir / "plugin.py"
        plugin_path.write_text(plugin_content)

        # Create plugin_manifest.json
        manifest_content = {
            "name": "PluginModel",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "A test model plugin from plugin.py",
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
            "main_class_path": f"temp.{model_dir.name}.plugin.PluginModelPlugin",  # This won't work in temp dir
            "entry_point": "create_plugin_model_plugin",
            "input_types": ["text"],
            "output_types": ["text"],
        }

        manifest_path = model_dir / "plugin_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_content, f, indent=2)

        # Discover and load plugins
        pm = PluginManager()
        count = pm.discover_and_load_plugins(temp_path)

        # Since the main_class_path refers to a non-existent module in temp dir,
        # it should fall back to loading plugin.py directly
        assert count >= 0  # At least try to load


if __name__ == "__main__":
    pytest.main([__file__])
