"""
Helper functions for plugin management tests.
"""

import json
from pathlib import Path


def create_mock_plugin_structure(base_path: Path):
    """Create a mock plugin structure for testing."""

    # Create a mock model directory
    model_dir = base_path / "test_model"
    model_dir.mkdir(parents=True)

    # Create plugin.py with a realistic test plugin
    plugin_py_content = """
from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface
)
from datetime import datetime
import torch
import torch.nn as nn

class TestModelPlugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="TestModel",
            version="1.0.0",
            author="Test Author",
            description="A test model plugin",
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
            return f"Processed: {data}"
        elif isinstance(data, (list, tuple)):
            return [f"Processed: {item}" for item in data]
        else:
            return f"Processed: {str(data)}"

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
        return f"{prompt} [GENERATED TEXT]"


def create_test_model_plugin():
    return TestModelPlugin()


__all__ = ["TestModelPlugin", "create_test_model_plugin"]
"""

    plugin_py_path = model_dir / "plugin.py"
    plugin_py_path.write_text(plugin_py_content)

    # Create plugin_manifest.json
    manifest_content = {
        "name": "TestModel",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "A test model plugin",
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
        "main_class_path": "src.inference_pio.models.test_model.plugin.TestModelPlugin",
        "entry_point": "create_test_model_plugin",
        "input_types": ["text"],
        "output_types": ["text"],
    }

    manifest_path = model_dir / "plugin_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_content, f, indent=2)

    # Create __init__.py
    init_path = model_dir / "__init__.py"
    init_path.write_text("# Test model init file")
