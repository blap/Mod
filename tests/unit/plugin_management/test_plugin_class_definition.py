"""
Test plugin class definition for plugin management tests.
"""

from datetime import datetime

import torch
import torch.nn as nn

from src.common.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface,
)


class RealisticTestPlugin(TextModelPluginInterface):
    """Realistic test plugin that mimics actual plugin behavior."""

    def __init__(self, metadata=None):
        if metadata is None:
            metadata = PluginMetadata(
                name="TestPlugin",
                version="1.0.0",
                author="Test Author",
                description="A test plugin",
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
        # Simulate actual initialization logic
        self._initialized = True
        return True

    def load_model(self, config=None):
        # Return a simple torch model instead of MagicMock
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        self._model = SimpleModel()
        return self._model

    def infer(self, data):
        # Simulate actual inference
        if not self._initialized:
            self.initialize()

        if isinstance(data, str):
            # Process string input
            return f"Processed: {data}"
        elif isinstance(data, (list, tuple)):
            # Process list/tuple input
            return [f"Processed: {item}" for item in data]
        else:
            # Process other types
            return f"Processed: {str(data)}"

    def cleanup(self) -> bool:
        # Simulate actual cleanup
        self._model = None
        self._initialized = False
        return True

    def supports_config(self, config) -> bool:
        # Simulate actual config validation
        return config is None or isinstance(config, dict)

    def tokenize(self, text: str, **kwargs):
        # Simulate actual tokenization
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        # Simple tokenization: split by spaces and convert to token IDs
        tokens = text.split()
        # Map to simple integer IDs (in real implementation, this would use actual tokenizer)
        token_map = {word: idx + 1 for idx, word in enumerate(set(tokens))}
        return [token_map[word] for word in tokens]

    def detokenize(self, token_ids, **kwargs) -> str:
        # Simulate actual detokenization
        if isinstance(token_ids, (list, tuple)):
            # Map token IDs back to words (simplified)
            # In real implementation, this would use actual vocabulary
            return " ".join([f"token_{tid}" for tid in token_ids])
        else:
            return f"token_{tid}"

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        # Simulate actual text generation
        if not self._initialized:
            self.initialize()

        # Simple generation logic
        return f"{prompt} [GENERATED TEXT]"
