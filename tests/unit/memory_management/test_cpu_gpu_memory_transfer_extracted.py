
import unittest
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath("src"))

from qwen3_vl.memory_management.cpu_gpu_memory_transfer_optimization import (
    MemoryTransferOptimizer,
    TransferType
)

class TestMemoryTransferOptimizer(unittest.TestCase):
    def test_transfer_optimizer(self):
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU transfer tests")
            return

        # Create optimizer
        optimizer = MemoryTransferOptimizer()

        # Test transfer
        tensor = torch.randn(100, 100)
        transferred = optimizer.transfer(tensor, TransferType.CPU_TO_GPU)
        self.assertTrue(transferred.is_cuda)

        back = optimizer.transfer(transferred, TransferType.GPU_TO_CPU)
        self.assertFalse(back.is_cuda)

if __name__ == "__main__":
    unittest.main()
