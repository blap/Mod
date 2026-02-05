"""
Solution to eliminate cross-dependencies between models in the Inference-PIO system.

Current Issue:
- The model adapter factory function in src/inference_pio/common/interfaces/model_adapter.py
  has a direct import from a specific model directory:
  `from ..models.qwen3_vl_2b.qwen3_vl_2b_model_adapter import Qwen3VL2BModelAdapter`
  
This creates a hard dependency between the common interface and a specific model implementation,
violating the principle that models should be independent and share components only through
common interfaces.

Solution:
1. Create a registry system for model adapters
2. Remove direct imports between models
3. Use dynamic loading or registration pattern
"""

import logging
from typing import Any, Dict, Optional, Type
import importlib

logger = logging.getLogger(__name__)

# Registry to store model adapter classes by model type
MODEL_ADAPTER_REGISTRY: Dict[str, Type] = {}

def register_model_adapter(model_type: str, adapter_class: Type):
    """
    Register a model adapter class for a specific model type.
    
    Args:
        model_type: String identifier for the model type
        adapter_class: The model adapter class to register
    """
    MODEL_ADAPTER_REGISTRY[model_type] = adapter_class
    logger.info(f"Registered model adapter for {model_type}: {adapter_class.__name__}")

def get_registered_model_adapter(model_type: str):
    """
    Get a registered model adapter class by model type.
    
    Args:
        model_type: String identifier for the model type
        
    Returns:
        The registered model adapter class, or None if not found
    """
    return MODEL_ADAPTER_REGISTRY.get(model_type)

def get_model_adapter_by_type(model_type: str):
    """
    Dynamically load and return the appropriate model adapter based on model type.
    
    Args:
        model_type: String identifier for the model type
        
    Returns:
        An instance of the appropriate model adapter
    """
    # First, try to get from registry
    adapter_class = get_registered_model_adapter(model_type)
    if adapter_class:
        return adapter_class
    
    # If not in registry, try dynamic import
    adapter_mapping = {
        'glm47': 'src.inference_pio.models.glm_4_7_flash.model_adapter',
        'qwen3_4b_instruct_2507': 'src.inference_pio.models.qwen3_4b_instruct_2507.model_adapter',
        'qwen3_coder_30b': 'src.inference_pio.models.qwen3_coder_30b.model_adapter',
        'qwen3_vl_2b': 'src.inference_pio.models.qwen3_vl_2b.qwen3_vl_2b_model_adapter',
        'qwen3_coder_next': 'src.inference_pio.models.qwen3_coder_next.model_adapter',
        'qwen3_0_6b': 'src.inference_pio.models.qwen3_0_6b.model_adapter'
    }
    
    module_path = adapter_mapping.get(model_type)
    if module_path:
        try:
            module = importlib.import_module(module_path)
            # Assuming the adapter class has a predictable name
            adapter_attr_name = f"{model_type.replace(' ', '_').replace('-', '_').title()}ModelAdapter"
            if hasattr(module, adapter_attr_name):
                return getattr(module, adapter_attr_name)
            else:
                # Try to find any class ending with 'ModelAdapter'
                for attr_name in dir(module):
                    if attr_name.endswith('ModelAdapter') and attr_name != 'BaseModelAdapter':
                        return getattr(module, attr_name)
        except ImportError as e:
            logger.warning(f"Could not import model adapter for {model_type}: {e}")
    
    return None

def identify_model_type(model: Any) -> str:
    """
    Identify the model type based on the model instance.
    
    Args:
        model: The model instance to identify
        
    Returns:
        String identifier for the model type
    """
    model_str = str(type(model)).lower()
    
    if "glm" in model_str and "4" in model_str and "flash" in model_str:
        return "glm47"
    elif "qwen3" in model_str and "4b" in model_str and "instruct" in model_str:
        return "qwen3_4b_instruct_2507"
    elif "qwen3" in model_str and ("coder" in model_str and "30b" in model_str.lower()):
        return "qwen3_coder_30b"
    elif "qwen3" in model_str and ("vl" in model_str and "2b" in model_str.lower()):
        return "qwen3_vl_2b"
    elif "qwen3" in model_str and "coder" in model_str and "next" in model_str:
        return "qwen3_coder_next"
    elif "qwen3" in model_str and "0.6b" in model_str:
        return "qwen3_0_6b"
    else:
        # Return a generic identifier based on class name
        class_name = type(model).__name__.lower()
        if "glm" in class_name:
            return "glm47"
        elif "qwen3" in class_name:
            return "qwen3_generic"
        else:
            return "generic"

def get_model_adapter(model: Any, nas_controller: Any):
    """
    Refactored function to get the appropriate model adapter without cross-dependencies.
    
    Args:
        model: The model instance
        nas_controller: The NAS controller instance
        
    Returns:
        An instance of the appropriate model adapter
    """
    model_type = identify_model_type(model)
    
    # Try to get from registry first
    adapter_class = get_registered_model_adapter(model_type)
    
    # If not found in registry, try dynamic loading
    if not adapter_class:
        adapter_class = get_model_adapter_by_type(model_type)
        
    # If still not found, fall back to base adapter
    if not adapter_class:
        logger.warning(f"Unknown model type '{model_type}', using basic adapter")
        from src.inference_pio.common.interfaces.model_adapter import BaseModelAdapter
        return BaseModelAdapter(model, nas_controller)
    
    # Create and return an instance of the adapter
    return adapter_class(model, nas_controller)

# Registration functions for each model type (these would be called when each model is loaded)
def register_all_model_adapters():
    """
    Register all known model adapters. This function would typically be called 
    during system initialization.
    """
    # This is a placeholder - in practice, each model would register its adapter
    # when it's loaded or initialized
    raise NotImplementedError("Method not implemented")

# Example of how each model would register its adapter:
def register_glm47_model_adapter(adapter_class):
    register_model_adapter('glm47', adapter_class)

def register_qwen3_4b_instruct_2507_model_adapter(adapter_class):
    register_model_adapter('qwen3_4b_instruct_2507', adapter_class)

def register_qwen3_coder_30b_model_adapter(adapter_class):
    register_model_adapter('qwen3_coder_30b', adapter_class)

def register_qwen3_vl_2b_model_adapter(adapter_class):
    register_model_adapter('qwen3_vl_2b', adapter_class)

def register_qwen3_coder_next_model_adapter(adapter_class):
    register_model_adapter('qwen3_coder_next', adapter_class)

def register_qwen3_0_6b_model_adapter(adapter_class):
    register_model_adapter('qwen3_0_6b', adapter_class)

print("Cross-dependency elimination solution created.")
print("Key changes needed:")
print("1. Remove direct import from common interface: 'from ..models.qwen3_vl_2b.qwen3_vl_2b_model_adapter import Qwen3VL2BModelAdapter'")
print("2. Use registry or dynamic loading pattern instead")
print("3. Each model should register its adapter when loaded")
print("4. Common interface should only depend on base classes and interfaces")