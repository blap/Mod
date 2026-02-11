
import sys
import os

# Adjust path so we can import src
sys.path.append(os.getcwd())

from src.inference_pio.core.engine.backend import Linear, Tensor, Module

def test_module_to_fix():
    print("Testing Module.to fix...")
    l = Linear(10, 10)
    original_weight = l.weight

    print(f"Original weight ID: {id(original_weight)}")

    # Trigger move to fake CUDA device (force creation of new tensor even if fallback to CPU)
    # We use "cuda:0" which is != "cpu" string check in .to()
    l.to("cuda:0")

    new_weight = l.weight
    print(f"New weight ID: {id(new_weight)}")

    if original_weight is new_weight:
        print("FAIL: weight attribute was not updated!")
    else:
        print("PASS: weight attribute was updated.")

    if l._parameters['weight'] is new_weight:
        print("PASS: _parameters['weight'] matches attribute.")
    else:
        print("FAIL: _parameters['weight'] does not match attribute!")

if __name__ == "__main__":
    try:
        test_module_to_fix()
    except Exception as e:
        print(f"Error: {e}")
