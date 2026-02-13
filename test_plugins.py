"""
Test script to verify all 6 model plugins work correctly
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PluginTester")

# Adicionando o caminho para os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_plugin(name, module_path, class_name):
    logger.info(f"Testing {name} Plugin...")
    try:
        # Import dynamically
        module = __import__(module_path, fromlist=[class_name])
        PluginClass = getattr(module, class_name)
        
        # Instantiate
        plugin = PluginClass()
        
        # Initialize (with dummy path to trigger SimpleTokenizer fallback if needed)
        success = plugin.initialize(model_path="/tmp/test_dummy")

        # Checks
        if not success:
            logger.error(f"Plugin {name} failed to initialize.")
            return False

        if not hasattr(plugin, '_model') or not plugin._model:
            logger.error(f"Plugin {name} _model is None.")
            return False

        if not hasattr(plugin, 'tokenizer'):
            logger.error(f"Plugin {name} has no 'tokenizer' attribute.")
            return False

        # Basic Interface Check
        info = plugin.get_model_info() if hasattr(plugin, 'get_model_info') else "No info"
        logger.info(f"{name} Initialized. Info: {info}")

        return True
        
    except Exception as e:
        logger.error(f"Error testing {name} plugin: {e}")
        return False

def main():
    logger.info("Starting comprehensive plugin tests...")
    
    plugins_to_test = [
        ("Qwen3-Coder-Next", "src.inference_pio.models.qwen3_coder_next.plugin", "Qwen3CoderNextPlugin"),
        ("GLM-4.7-Flash", "src.inference_pio.models.glm_4_7_flash.plugin", "GLM_4_7_Flash_Plugin"),
        ("Qwen3-VL-2B", "src.inference_pio.models.qwen3_vl_2b.plugin", "Qwen3_VL_2B_Plugin"),
        ("Qwen3-4B-Instruct-2507", "src.inference_pio.models.qwen3_4b_instruct_2507.plugin", "Qwen3_4B_Instruct_2507_Plugin"),
        ("Qwen3-0.6B", "src.inference_pio.models.qwen3_0_6b.plugin", "Qwen3_0_6B_Plugin"),
        ("Qwen3-Coder-30B", "src.inference_pio.models.qwen3_coder_30b.plugin", "Qwen3_Coder_30B_Plugin"),
    ]
    
    results = []
    for name, path, cls in plugins_to_test:
        res = test_plugin(name, path, cls)
        results.append(res)

    if all(results):
        logger.info("All 6 plugins verified successfully!")
        sys.exit(0)
    else:
        logger.error("Some plugins failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
