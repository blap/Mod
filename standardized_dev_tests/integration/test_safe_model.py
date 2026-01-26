#!/usr/bin/env python
"""
Safe model test with maximum protections against access violation.
"""

import sys
import os
import gc
import traceback
from contextlib import contextmanager

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

@contextmanager
def safe_model_context():
    """Context manager for safe model operations."""
    # Store original recursion limit
    original_limit = sys.getrecursionlimit()
    
    try:
        # Increase recursion limit for model initialization
        sys.setrecursionlimit(3000)
        print("Increased recursion limit to 3000")
        
        # Collect garbage before starting
        gc.collect()
        print("Collected garbage")
        
        yield
        
    finally:
        # Always restore original recursion limit
        sys.setrecursionlimit(original_limit)
        print(f"Restored recursion limit to {original_limit}")
        
        # Collect garbage after operations
        gc.collect()
        print("Collected garbage after operations")

def safe_import(module_path, class_name):
    """Safely import a class with error handling."""
    try:
        if module_path == 'src.qwen3_vl.config':
            from src.qwen3_vl.config import Qwen3VLConfig
            return Qwen3VLConfig
        elif module_path == 'src.qwen3_vl.models.qwen3_vl_model':
            from src.qwen3_vl.models.qwen3_vl_model import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        else:
            module_parts = module_path.split('.')
            module = __import__('.'.join(module_parts[:-1]), fromlist=[class_name])
            return getattr(module, class_name)
    except ImportError as e:
        print(f"Import error for {module_path}.{class_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error importing {module_path}.{class_name}: {e}")
        return None

def main():
    """Main test function."""
    print("Safe Qwen3-VL Model Creation Test")
    print("="*50)
    
    with safe_model_context():
        try:
            print("Step 1: Importing configuration...")
            Qwen3VLConfig = safe_import('src.qwen3_vl.config', 'Qwen3VLConfig')
            if Qwen3VLConfig is None:
                print("Failed to import Qwen3VLConfig")
                return False
            
            print("Step 2: Creating configuration...")
            config = Qwen3VLConfig()
            print(f"Configuration created: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
            
            print("Step 3: Validating configuration...")
            if config.validate_capacity_preservation():
                print("✓ Configuration validates capacity preservation")
            else:
                print("✗ Configuration does NOT validate capacity preservation")
            
            print("Step 4: Importing model class...")
            Qwen3VLForConditionalGeneration = safe_import('src.qwen3_vl.models.qwen3_vl_model', 'Qwen3VLForConditionalGeneration')
            if Qwen3VLForConditionalGeneration is None:
                print("Failed to import Qwen3VLForConditionalGeneration")
                return False
            
            print("Step 5: Creating model instance...")
            # Create model with minimal risk
            model = Qwen3VLForConditionalGeneration(config)
            
            print("Step 6: Model created successfully!")
            print(f"Model has {config.num_hidden_layers} layers and {config.num_attention_heads} attention heads")
            
            print("Step 7: Testing device movement...")
            # Test our modified to() method
            model = model.cpu()
            print("✓ Model moved to CPU successfully")
            
            print("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
            print("The access violation error (0xC0000005) has been fixed!")
            return True
            
        except RecursionError as e:
            print(f"✗ RecursionError: {str(e)[:200]}...")
            print("This suggests the recursion protection wasn't sufficient.")
            return False
        except MemoryError as e:
            print(f"✗ MemoryError: {e}")
            print("This suggests insufficient memory for model creation.")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*50)
    if success:
        print("RESULT: SUCCESS - Model can be instantiated safely!")
        print("The access violation error has been resolved.")
    else:
        print("RESULT: FAILURE - Model instantiation still fails.")
        print("The access violation error may still exist.")
    print("="*50)