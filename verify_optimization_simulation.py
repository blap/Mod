import torch
import torch.nn as nn
from dataclasses import dataclass
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from src.qwen3_vl.memory_management.optimized_memory_management import (
    MemoryManager, optimize_model_memory, get_memory_manager, MemoryConfig
)
from src.qwen3_vl.optimization.hardware_optimizer import HardwareOptimizer

@dataclass
class DummyConfig:
    use_memory_efficient_attention: bool = False
    hardware_compute_capability: tuple = (6, 1)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def gradient_checkpointing_enable(self):
        print("Gradient checkpointing enabled on DummyModel.")

def verify_simulation():
    print("Verifying Optimization Logic (Simulation)...")

    # 1. Config & Manager
    config = MemoryConfig(memory_pool_size=1024*1024)
    manager = get_memory_manager(config)
    print("Memory Manager Initialized.")

    # 2. Model
    model = DummyModel()
    print("Dummy Model Created.")

    # 3. Optimize Memory
    optimize_model_memory(model, manager, config)
    print("Memory Optimizations Applied.")

    # 4. Hardware Optimization
    hw_opt = HardwareOptimizer()
    model = hw_opt.optimize_model(model)
    print("Hardware Optimizations Applied.")

    # Check results
    success = True
    if model.config.use_memory_efficient_attention:
        print("CHECK PASSED: Memory Efficient Attention Flag Set.")
    else:
        print("CHECK FAILED: Memory Efficient Attention Flag NOT Set.")
        success = False

    if success:
        print("\nOptimization Verification: SUCCESS")
    else:
        print("\nOptimization Verification: FAILURE")
        sys.exit(1)

if __name__ == "__main__":
    verify_simulation()
