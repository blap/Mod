"""
Simple test to verify that the test infrastructure works correctly.
"""

import sys
from pathlib import Path

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from inference_pio.common.standard_plugin_interface import PluginType
from inference_pio.test_utils import assert_equal, assert_true, run_tests


def test_basic_functionality():
    """Test basic functionality to verify test setup."""
    # Test that we can access basic enums/classes
    assert_equal(PluginType.MODEL_COMPONENT.value, "model_component")
    assert_true(True)


if __name__ == '__main__':
    run_tests([test_basic_functionality])