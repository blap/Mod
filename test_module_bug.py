
import sys
import os

# Adjust path so we can import src
sys.path.append(os.getcwd())

from src.inference_pio.core.engine.backend import Linear, Tensor, Module

def test_module_to():
    l = Linear(10, 10)
    print(f"Original device: {l.weight.device}")

    # Check if weight and _parameters['weight'] are same object
    print(f"Same object before: {l.weight is l._parameters['weight']}")

    l.to("cpu") # executing .to() on same device returns self, so objects remain same

    # Simulate move (even if fake)
    # We need to mock .to() to return a new object to test the reference update issue
    # But Tensor.to returns new object if device is different.

    # Let's try to mock a fake device transfer if we don't have cuda
    # Or just rely on the fact that to() returns a new tensor

    # Let's force a new tensor creation by manually doing what .to does
    new_t = Tensor(list(l.weight.shape), device="cuda") # Fake cuda device string

    # Manually reproduce what Module.to does
    l._parameters['weight'] = new_t

    print(f"After manual dict update:")
    print(f"Dictionary points to device: {l._parameters['weight'].device}")
    print(f"Attribute points to device: {l.weight.device}")
    print(f"Same object after: {l.weight is l._parameters['weight']}")

if __name__ == "__main__":
    try:
        test_module_to()
    except Exception as e:
        print(f"Error: {e}")
