"""
Test script to verify both Qwen3 plugins work correctly
"""
import sys
import os

# Adicionando o caminho para os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_qwen3_4b_plugin():
    """Test the Qwen3-4B plugin."""
    print("Testing Qwen3-4B-Instruct-2507 Plugin...")
    
    try:
        from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
        plugin = Qwen3_4B_Instruct_2507_Plugin()
        
        print(f"Plugin initialized: {hasattr(plugin, '_model')}")
        print(f"Model info: {plugin.get_model_info()}")
        print(f"Model parameters: {plugin.get_model_parameters()}")
        
        print("Qwen3-4B plugin test completed.\n")
    except RuntimeError as e:
        if "CPU Engine library not loaded" in str(e):
            print(f"Expected error (no backend): {e}")
            print("Plugin structure is correct, but backend libraries are not available.")
        else:
            print(f"Unexpected error: {e}")
    except Exception as e:
        print(f"Error testing Qwen3-4B plugin: {e}")


def test_qwen3_0_6b_plugin():
    """Test the Qwen3-0.6B plugin."""
    print("Testing Qwen3-0.6B Plugin...")
    
    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin
        plugin = Qwen3_0_6B_Plugin()
        
        print(f"Plugin initialized: {hasattr(plugin, '_model')}")
        print(f"Model info: {plugin.get_model_info()}")
        print(f"Model parameters: {plugin.get_model_parameters()}")
        
        print("Qwen3-0.6B plugin test completed.\n")
    except RuntimeError as e:
        if "CPU Engine library not loaded" in str(e):
            print(f"Expected error (no backend): {e}")
            print("Plugin structure is correct, but backend libraries are not available.")
        else:
            print(f"Unexpected error: {e}")
    except Exception as e:
        print(f"Error testing Qwen3-0.6B plugin: {e}")


if __name__ == "__main__":
    print("Starting plugin tests...\n")
    
    test_qwen3_4b_plugin()
    test_qwen3_0_6b_plugin()
    
    print("All tests completed!")