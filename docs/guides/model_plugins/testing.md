# Testing Model Plugins

## Step 8: Test the Plugin

Each model **MUST** have its own independent test suite located in `tests/` within the model directory. These tests should not rely on other models or shared test state that isn't provided by the standard test infrastructure.

### Directory Structure
```
src/inference_pio/models/model_name/tests/
├── unit/           # Unit tests for components
├── integration/    # Integration tests for the full plugin
└── performance/    # Performance benchmarks
```

### Writing Tests

Create tests to ensure the plugin works correctly:

```python
# In src/inference_pio/models/model_name/tests/unit/test_plugin.py
import unittest
from src.inference_pio.models.model_name.plugin import create_model_name_plugin

class TestModelName(unittest.TestCase):
    """
    Tests for the ModelName model.
    """
    def test_create_plugin(self):
        """
        Tests ModelName plugin creation.
        """
        plugin = create_model_name_plugin()
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.metadata.name, "ModelName")

    def test_initialize_plugin(self):
        """
        Tests ModelName plugin initialization.
        """
        plugin = create_model_name_plugin()
        # Mocking initialization to avoid loading weights in unit tests is recommended
        # result = plugin.initialize()
        # self.assertTrue(result)
```

## Benchmarks

Each model should also have its own benchmarks in `benchmarks/`.

```
src/inference_pio/models/model_name/benchmarks/
├── benchmark_inference_speed.py
└── benchmark_memory_usage.py
```

## Final Considerations

-   **Independence**: Tests for Model A must pass even if Model B is broken or missing.
-   **Structure**: Follow the directory structure strictly.
-   **Coverage**: Aim for high coverage of the `plugin.py` and `model.py` logic.
