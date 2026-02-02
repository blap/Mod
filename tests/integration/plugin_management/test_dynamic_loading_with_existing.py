"""
Test to demonstrate the dynamic loading system with the existing Qwen3 plugin
"""

import os
import sys

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import pytest

from src.plugins.manager import PluginManager, discover_and_load_plugins


def test_dynamic_loading_with_existing_qwen3_plugin():
    """Test the dynamic loading system with the existing Qwen3-0.6B plugin."""

    # Create plugin manager and discover plugins from the existing models directory
    pm = PluginManager()

    # Discover and load plugins from the existing models directory
    models_dir = Path("src/models")
    count = pm.discover_and_load_plugins(models_dir)

    # Verify that at least the Qwen3-0.6B plugin was loaded
    assert count >= 1, f"Expected at least 1 plugin to be loaded, got {count}"
    assert (
        len(pm.plugins) >= 1
    ), f"Expected at least 1 plugin in registry, got {len(pm.plugins)}"

    # Check if Qwen3-0.6B plugin is loaded
    if "Qwen3-0.6B" in pm.plugins:
        qwen3_plugin = pm.plugins["Qwen3-0.6B"]
        assert qwen3_plugin.metadata.name == "Qwen3-0.6B"
        assert qwen3_plugin.metadata.version == "1.0.0"
        assert qwen3_plugin.metadata.author == "Alibaba Cloud"

        print(f"Successfully loaded {len(pm.plugins)} plugins")
        print(f"Available plugins: {list(pm.plugins.keys())}")
    else:
        print(f"Available plugins: {list(pm.plugins.keys())}")
        # If Qwen3-0.6B is not found, check if any other plugin was loaded
        assert (
            len(pm.plugins) > 0
        ), "At least one plugin should be loaded from the models directory"


def test_dynamic_loading_methods_directly():
    """Test the individual dynamic loading methods directly."""
    from pathlib import Path

    from src.plugins.manager import PluginManager

    pm = PluginManager()

    # Test loading the Qwen3 plugin directly from its manifest
    qwen3_dir = Path("src/models/qwen3_0_6b")
    manifest_path = qwen3_dir / "plugin_manifest.json"

    if manifest_path.exists():
        # Load plugin from manifest
        loaded_count = pm._load_plugin_from_manifest(qwen3_dir, manifest_path)
        assert (
            loaded_count == 1
        ), f"Expected to load 1 plugin from manifest, got {loaded_count}"

        # Check that the plugin was loaded
        assert "Qwen3-0.6B" in pm.plugins
        plugin = pm.plugins["Qwen3-0.6B"]
        assert plugin.metadata.name == "Qwen3-0.6B"

        print(f"Direct manifest loading successful: {plugin.metadata.name}")
    else:
        print("Qwen3 manifest not found, skipping direct test")


if __name__ == "__main__":
    test_dynamic_loading_with_existing_qwen3_plugin()
    test_dynamic_loading_methods_directly()
    print("All dynamic loading tests with existing plugins passed!")
