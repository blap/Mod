# Model Plugin Implementation

## Step 4: Create the Model Implementation

Create a `model.py` file with the model implementation using the custom backend. **Do not use PyTorch or NumPy.**

```python
import logging
from .config import ModelNameConfig
from ...core.engine.backend import Module, Tensor, Linear, RMSNorm, scaled_dot_product_attention

logger = logging.getLogger(__name__)

class ModelNameModel(Module):
    """
    Implementation of the ModelName model using C-Engine.
    """
    def __init__(self, config: ModelNameConfig):
        super().__init__()
        self.config = config
        # Initialize model components using backend Modules
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor, **kwargs):
        # Implement forward pass using Tensor operations
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def generate(self, input_ids: Tensor, max_new_tokens: int = 100):
        # Implement autoregressive generation loop
        pass
```

## Step 5: Create the Plugin

Create a `plugin.py` file. The class must implement `TextModelPluginInterface`.

### Minimum Implementation Requirements

You **must** implement the following methods according to the standard interface:
*   `initialize(**kwargs)`: Setup resources and `BatchManager`.
*   `load_model(config)`: Load weights (using `safetensors`).
*   `tokenize(text)`: Convert string to `List[float]`. **Must be robust (try/except).**
*   `detokenize(token_ids)`: Convert `List[int]` back to string. **Must be robust.**
*   `infer_batch(requests)`: Handle batch requests via `BatchManager`.
*   `generate_text(prompt, ...)`: High-level generation entry point.
*   `cleanup()`: Release memory.

### Example Implementation (Text Model)

```python
import logging
from typing import Any, List, Union
from ...common.interfaces.improved_base_plugin_interface import (
    TextModelPluginInterface,
    PluginMetadata,
    PluginType
)
from .config import ModelNameConfig
from .model import ModelNameModel
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class ModelName_Plugin(TextModelPluginInterface):
    """
    Plugin for the ModelName model.
    """
    def __init__(self):
        metadata = PluginMetadata(
            name="ModelName",
            version="1.0.0",
            author="Your Name",
            description="Plugin for ModelName",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[], # No external dependencies
            compatibility={
                "python_version": ">=3.8",
                "min_memory_gb": 8.0
            },
            model_architecture="Transformer",
            model_size="7B",
            required_memory_gb=16.0,
            supported_modalities=["text"],
        )
        super().__init__(metadata)
        self._config = ModelNameConfig()
        self._model = None
        self.batch_manager = None

    def initialize(self, **kwargs) -> bool:
        try:
            # Update config
            for k, v in kwargs.items():
                if hasattr(self._config, k): setattr(self._config, k, v)

            logger.info("Initializing ModelName plugin...")
            self.load_model()

            # Initialize Standard Batch Manager
            from ...common.managers.batch_manager import BatchManager
            self.batch_manager = BatchManager(self._model)

            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def load_model(self, config=None):
        if config: self._config = config
        self._model = ModelNameModel(self._config)
        # Load weights logic here (e.g., ModelLoader)
        return self._model

    def infer(self, data: Any) -> Any:
        if isinstance(data, str):
            return self.generate_text(data)
        return ""

    def tokenize(self, text: str, **kwargs) -> List[float]:
        # Robust tokenization with fallback
        if hasattr(self._model, 'tokenizer') and self._model.tokenizer:
            try:
                return [float(x) for x in self._model.tokenizer.encode(text)]
            except: pass
        return [1.0] * 5 # Dummy fallback

    def detokenize(self, token_ids: List[int], **kwargs) -> str:
        # Robust detokenization
        if hasattr(self._model, 'tokenizer') and self._model.tokenizer:
            try:
                return self._model.tokenizer.decode(token_ids)
            except: pass
        return f"Generated {len(token_ids)} tokens"

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        results = []
        if not self.batch_manager: return super().infer_batch(requests)

        # Add requests to batch manager
        req_ids = []
        for i, prompt in enumerate(requests):
            ids = self.tokenize(prompt)
            rid = 1000 + i # Unique ID generation
            self.batch_manager.add_request(rid, ids)
            req_ids.append(rid)

        # Step through batch
        for _ in req_ids:
            out_tensor = self.batch_manager.step()
            if out_tensor:
                res = self.detokenize([int(x) for x in out_tensor.to_list()])
                results.append(res)
            else:
                results.append("Error")
        return results

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._model: self.load_model()

        try:
            ids = self.tokenize(prompt)
            t = Tensor([1, len(ids)])
            t.load(ids)

            out = self._model.generate(t, max_new_tokens=max_new_tokens)
            return self.detokenize([int(x) for x in out.to_list()])
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Error"

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_model_name_plugin() -> ModelName_Plugin:
    return ModelName_Plugin()
```

## Step 6: Update `__init__.py`

Update `__init__.py` to export components.

## Checklist

*   [ ] **No PyTorch/NumPy usage.**
*   [ ] **Inherits `TextModelPluginInterface`.**
*   [ ] **Implements `tokenize`/`detokenize` with Try/Except.**
*   [ ] **Implements `infer_batch` using `BatchManager`.**
*   [ ] **Uses `backend.Tensor` for all math.**
