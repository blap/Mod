#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify that the access violation fix works.
"""

import torch
import sys
import os

# Adicionando o caminho do projeto para importações
sys.path.insert(0, os.path.join(os.getcwd()))

def test_fixed_model():
    """Test the fixed model initialization."""
    print("Testing fixed Qwen3-VL model initialization...")
    
    try:
        print("1. Importing configuration...")
        from src.qwen3_vl.config import Qwen3VLConfig
        
        print("2. Creating configuration...")
        config = Qwen3VLConfig()
        # Usar configuração reduzida para testes iniciais
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.hidden_size = 128
        config.vision_num_hidden_layers = 2
        config.vision_num_attention_heads = 2
        config.vision_hidden_size = 64
        
        print(f"   Config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
        
        print("3. Validating configuration...")
        if config.validate_capacity_preservation():
            print("   ✓ Configuration validated")
        else:
            print("   ⚠ Configuration validation failed")
        
        print("4. Importing model class...")
        from src.qwen3_vl.models import Qwen3VLForConditionalGeneration
        
        print("5. Creating model instance...")
        model = Qwen3VLForConditionalGeneration(config)
        print("   ✓ Model created successfully")
        
        print("6. Moving model to CPU...")
        model = model.cpu()
        print("   ✓ Model moved to CPU successfully")
        
        print("7. Testing forward pass...")
        # Criar entradas pequenas para o teste
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        pixel_values = torch.randn(1, 3, 64, 64)  # Imagem pequena
        
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
        
        print(f"   ✓ Forward pass completed successfully")
        print(f"   Output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"   Output keys: {list(output.keys())}")
        
        print("\n🎉 SUCCESS: Model initialization and basic operation completed without access violation!")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 70)
    print("Qwen3-VL Access Violation Fix Verification Test")
    print("=" * 70)
    
    success = test_fixed_model()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ VERIFICATION PASSED: The access violation fix is working!")
        print("   The model can now be instantiated without causing access violations.")
    else:
        print("❌ VERIFICATION FAILED: The access violation may still occur.")
    print("=" * 70)

if __name__ == "__main__":
    main()