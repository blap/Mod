#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step-by-step test to identify where the access violation occurs in Qwen3-VL model initialization.
"""

import torch
import sys
import os
import traceback

# Adicionando o caminho do projeto para importações
sys.path.insert(0, os.path.join(os.getcwd()))

def test_step(step_name, test_func):
    """Helper function to run a test step with error handling."""
    print(f"\n--- Step: {step_name} ---")
    try:
        result = test_func()
        print(f"✅ {step_name}: PASSED")
        return True, result
    except KeyboardInterrupt:
        print(f"⚠ {step_name}: INTERRUPTED BY USER")
        return False, None
    except SystemExit:
        print(f"⚠ {step_name}: SYSTEM EXIT")
        return False, None
    except Exception as e:
        print(f"❌ {step_name}: FAILED - {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False, None

def step_import_config():
    """Step 1: Import configuration."""
    from src.qwen3_vl.config import Qwen3VLConfig
    print("Qwen3VLConfig imported successfully")
    return Qwen3VLConfig

def step_create_config():
    """Step 2: Create configuration."""
    from src.qwen3_vl.config import Qwen3VLConfig
    config = Qwen3VLConfig()
    # Use minimal configuration to avoid memory issues
    config.num_hidden_layers = 1
    config.num_attention_heads = 1
    config.hidden_size = 8
    config.vision_num_hidden_layers = 1
    config.vision_num_attention_heads = 1
    config.vision_hidden_size = 8
    print(f"Config created: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
    return config

def step_validate_config(config):
    """Step 3: Validate configuration."""
    is_valid = config.validate_capacity_preservation()
    print(f"Config validation: {is_valid}")
    return is_valid

def step_import_models():
    """Step 4: Import models module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "models", 
        "C:/Users/Admin/Documents/GitHub/Mod/src/qwen3_vl/models.py"
    )
    models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_module)
    print("Models module loaded successfully")
    return models_module

def step_get_model_class(models_module):
    """Step 5: Get model class."""
    Qwen3VLForConditionalGeneration = getattr(models_module, 'Qwen3VLForConditionalGeneration')
    print("Qwen3VLForConditionalGeneration class obtained")
    return Qwen3VLForConditionalGeneration

def step_create_model(Qwen3VLForConditionalGeneration, config):
    """Step 6: Create model instance."""
    model = Qwen3VLForConditionalGeneration(config)
    print("Model created successfully")
    return model

def step_move_to_cpu(model):
    """Step 7: Move model to CPU."""
    model = model.cpu()
    print("Model moved to CPU successfully")
    return model

def step_forward_pass(model, config):
    """Step 8: Test forward pass."""
    input_ids = torch.randint(0, config.vocab_size, (1, 2))
    pixel_values = torch.randn(1, 3, 32, 32)  # Small image for testing
    
    with torch.no_grad():
        output = model(input_ids=input_ids, pixel_values=pixel_values)
    
    print(f"Forward pass completed. Output type: {type(output)}")
    return output

def main():
    """Main function to run all steps."""
    print("=" * 60)
    print("Step-by-Step Qwen3-VL Access Violation Test")
    print("=" * 60)
    
    # Step 1: Import configuration
    success1, Qwen3VLConfig = test_step("Import Config", step_import_config)
    
    if not success1:
        print("\n❌ Test stopped at Step 1: Import Config")
        return
    
    # Step 2: Create configuration
    success2, config = test_step("Create Config", step_create_config)
    
    if not success2:
        print("\n❌ Test stopped at Step 2: Create Config")
        return
    
    # Step 3: Validate configuration
    success3, _ = test_step("Validate Config", lambda: step_validate_config(config))
    
    if not success3:
        print("\n❌ Test stopped at Step 3: Validate Config")
        return
    
    # Step 4: Import models module
    success4, models_module = test_step("Import Models", step_import_models)
    
    if not success4:
        print("\n❌ Test stopped at Step 4: Import Models")
        return
    
    # Step 5: Get model class
    success5, ModelClass = test_step("Get Model Class", lambda: step_get_model_class(models_module))
    
    if not success5:
        print("\n❌ Test stopped at Step 5: Get Model Class")
        return
    
    # Step 6: Create model
    success6, model = test_step("Create Model", lambda: step_create_model(ModelClass, config))
    
    if not success6:
        print("\n❌ Test stopped at Step 6: Create Model")
        return
    
    # Step 7: Move to CPU
    success7, model = test_step("Move to CPU", lambda: step_move_to_cpu(model))
    
    if not success7:
        print("\n❌ Test stopped at Step 7: Move to CPU")
        return
    
    # Step 8: Forward pass
    success8, output = test_step("Forward Pass", lambda: step_forward_pass(model, config))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"1. Import Config: {'✅' if success1 else '❌'}")
    print(f"2. Create Config: {'✅' if success2 else '❌'}")
    print(f"3. Validate Config: {'✅' if success3 else '❌'}")
    print(f"4. Import Models: {'✅' if success4 else '❌'}")
    print(f"5. Get Model Class: {'✅' if success5 else '❌'}")
    print(f"6. Create Model: {'✅' if success6 else '❌'}")
    print(f"7. Move to CPU: {'✅' if success7 else '❌'}")
    print(f"8. Forward Pass: {'✅' if success8 else '❌'}")
    
    if all([success1, success2, success3, success4, success5, success6, success7, success8]):
        print("\n🎉 All steps passed! No access violation detected.")
    else:
        print("\n💥 Some steps failed. Access violation likely occurred in the failed step.")
    print("=" * 60)

if __name__ == "__main__":
    main()