"""
Testes abrangentes para o sistema de pooling de tensores (MemoryPool).
"""

import unittest
import sys
import os
import torch
import numpy as np

# Adiciona o diretório pai ao path para permitir importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from memory_optimization_systems.tensor_pooling_system.memory_pool import MemoryPool, DataType, allocate_tensor, deallocate_tensor, get_global_memory_pool


class TestMemoryPool(unittest.TestCase):
    """Testes para a classe MemoryPool."""

    def setUp(self):
        """Configuração antes de cada teste."""
        self.pool = MemoryPool(max_pool_size_mb=10)  # 10MB pool for testing

    def test_initialization(self):
        """Testa a inicialização correta do pool de memória."""
        self.assertEqual(self.pool.max_pool_size_bytes, 10 * 1024 * 1024)
        self.assertEqual(self.pool.current_pool_size_bytes, 0)
        self.assertIsInstance(self.pool.pools, dict)
        self.assertEqual(len(self.pool.pools), 0)

    def test_get_tensor_basic(self):
        """Testa a obtenção básica de tensores do pool."""
        shape = (3, 224, 224)
        tensor = self.pool.get_tensor(shape, DataType.FLOAT32)
        
        self.assertEqual(tensor.shape, torch.Size(shape))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertFalse(tensor.requires_grad)

    def test_get_tensor_different_dtypes(self):
        """Testa a obtenção de tensores com diferentes tipos de dados."""
        shapes_and_dtypes = [
            ((10, 10), DataType.FLOAT32),
            ((5, 5), DataType.FLOAT16),
            ((20, 15), DataType.BFLOAT16),
            ((8, 8), DataType.INT32),
            ((12, 12), DataType.BOOL),
        ]
        
        for shape, dtype in shapes_and_dtypes:
            with self.subTest(shape=shape, dtype=dtype):
                tensor = self.pool.get_tensor(shape, dtype)
                self.assertEqual(tensor.shape, torch.Size(shape))
                self.assertEqual(tensor.dtype, dtype.value)

    def test_release_tensor(self):
        """Testa a liberação de tensores de volta ao pool."""
        shape = (4, 4)
        tensor = self.pool.get_tensor(shape, DataType.FLOAT32)
        
        initial_pool_size = self.pool.current_pool_size_bytes
        
        # Release the tensor back to the pool
        success = self.pool.release_tensor(tensor, shape)
        
        self.assertTrue(success)
        self.assertGreater(self.pool.current_pool_size_bytes, initial_pool_size)
        
        # Check that the tensor is now in the pool
        key = (shape, DataType.FLOAT32)
        self.assertIn(key, self.pool.pools)
        self.assertEqual(len(self.pool.pools[key]), 1)

    def test_tensor_reuse_from_pool(self):
        """Testa a reutilização de tensores do pool."""
        shape = (5, 5)
        
        # Get a tensor from the pool
        tensor1 = self.pool.get_tensor(shape, DataType.FLOAT32)
        
        # Release it back to the pool
        self.pool.release_tensor(tensor1, shape)
        
        # Get another tensor of the same shape - should come from the pool
        initial_stats = self.pool.get_stats()
        
        tensor2 = self.pool.get_tensor(shape, DataType.FLOAT32)
        
        final_stats = self.pool.get_stats()
        
        # Check that we had a pool hit
        self.assertEqual(final_stats['pool_hits'], initial_stats['pool_hits'] + 1)
        self.assertEqual(final_stats['allocations_from_pool'], initial_stats['allocations_from_pool'] + 1)
        
        # The tensor should have the correct properties
        self.assertEqual(tensor2.shape, torch.Size(shape))
        self.assertEqual(tensor2.dtype, torch.float32)

    def test_pool_size_limit(self):
        """Testa o limite de tamanho do pool."""
        # Create a very small pool for testing
        small_pool = MemoryPool(max_pool_size_mb=1)  # 1MB
        
        # Create a tensor that is larger than the pool size
        large_shape = (1000, 1000)  # Approximately 4MB for float32
        tensor = small_pool.get_tensor(large_shape, DataType.FLOAT32)
        
        # Try to release it back - should fail due to size limit
        success = small_pool.release_tensor(tensor, large_shape)
        
        self.assertFalse(success)
        self.assertEqual(small_pool.current_pool_size_bytes, 0)

    def test_direct_allocation_when_pool_full(self):
        """Testa alocação direta quando o pool está cheio."""
        # Fill the pool with many small tensors
        small_pool = MemoryPool(max_pool_size_mb=1)  # 1MB
        
        # Add tensors to fill the pool
        for i in range(50):  # 50 small tensors
            tensor = small_pool.get_tensor((10, 10), DataType.FLOAT32)
            # Don't release them back, so the pool fills up
        
        # Now try to get a tensor - should still work even though pool is conceptually full
        new_tensor = small_pool.get_tensor((5, 5), DataType.FLOAT32)
        
        # Should be allocated directly
        self.assertIsNotNone(new_tensor)
        self.assertEqual(new_tensor.shape, torch.Size((5, 5)))

    def test_compact_pool(self):
        """Testa a compactação do pool para reduzir fragmentação."""
        # Usar tensores maiores que 1KB para garantir que eles não sejam removidos
        shape1 = (50, 50)  # 50*50*4bytes = 10KB
        shape2 = (60, 60)  # 60*60*4bytes = 14.4KB

        # Add some tensors to the pool
        tensor1 = self.pool.get_tensor(shape1, DataType.FLOAT32)
        tensor2 = self.pool.get_tensor(shape2, DataType.FLOAT32)

        self.pool.release_tensor(tensor1, shape1)
        self.pool.release_tensor(tensor2, shape2)

        # Verify they're in the pool
        self.assertGreater(len(self.pool.pools[(shape1, DataType.FLOAT32)]), 0)
        self.assertGreater(len(self.pool.pools[(shape2, DataType.FLOAT32)]), 0)

        # Compact the pool
        self.pool.compact_pool()

        # Since our tensors are larger than 1KB threshold, they should remain in the pool
        self.assertIn((shape1, DataType.FLOAT32), self.pool.pools)
        self.assertIn((shape2, DataType.FLOAT32), self.pool.pools)

    def test_statistics_tracking(self):
        """Testa o rastreamento de estatísticas do pool."""
        initial_stats = self.pool.get_stats()
        
        # Perform some operations
        tensor1 = self.pool.get_tensor((10, 10), DataType.FLOAT32)
        self.pool.release_tensor(tensor1, (10, 10))
        tensor2 = self.pool.get_tensor((10, 10), DataType.FLOAT32)  # Should come from pool
        
        final_stats = self.pool.get_stats()
        
        # Check that statistics were updated correctly
        self.assertEqual(final_stats['allocations_from_pool'], initial_stats['allocations_from_pool'] + 1)
        self.assertEqual(final_stats['deallocations_to_pool'], initial_stats['deallocations_to_pool'] + 1)
        self.assertEqual(final_stats['pool_hits'], initial_stats['pool_hits'] + 1)

    def test_clear_pool(self):
        """Testa a limpeza completa do pool."""
        # Add some tensors to the pool
        tensor1 = self.pool.get_tensor((5, 5), DataType.FLOAT32)
        tensor2 = self.pool.get_tensor((10, 10), DataType.FLOAT16)
        
        self.pool.release_tensor(tensor1, (5, 5))
        self.pool.release_tensor(tensor2, (10, 10))
        
        # Verify pool has content
        stats_before = self.pool.get_stats()
        self.assertGreater(stats_before['current_pool_size_bytes'], 0)
        
        # Clear the pool
        clear_result = self.pool.clear_pool()
        
        # Verify pool is empty
        stats_after = self.pool.get_stats()
        self.assertEqual(stats_after['current_pool_size_bytes'], 0)
        self.assertEqual(len(self.pool.pools), 0)
        self.assertGreater(clear_result['total_tensors_cleared'], 0)

    def test_device_support(self):
        """Testa o suporte a diferentes dispositivos (CPU/GPU)."""
        # Test with CPU (default)
        tensor_cpu = self.pool.get_tensor((5, 5), DataType.FLOAT32, device='cpu')
        self.assertEqual(tensor_cpu.device.type, 'cpu')
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            tensor_cuda = self.pool.get_tensor((5, 5), DataType.FLOAT32, device='cuda')
            self.assertEqual(tensor_cuda.device.type, 'cuda')

    def test_torch_dtype_support(self):
        """Testa o suporte para tipos de dados torch diretamente."""
        # Test with torch.dtype instead of DataType enum
        tensor = self.pool.get_tensor((5, 5), torch.float16)
        self.assertEqual(tensor.dtype, torch.float16)
        
        # Test conversion for unsupported types
        tensor_double = self.pool.get_tensor((5, 5), torch.float64)
        self.assertEqual(tensor_double.dtype, torch.float32)  # Should convert to closest


class TestGlobalMemoryPool(unittest.TestCase):
    """Testes para o pool de memória global singleton."""

    def test_singleton_behavior(self):
        """Testa que o pool global é um singleton."""
        pool1 = get_global_memory_pool(5)
        pool2 = get_global_memory_pool(10)  # Different size should return same instance
        pool3 = get_global_memory_pool(5)   # Same size should return same instance
        
        self.assertIs(pool1, pool2)
        self.assertIs(pool1, pool3)
        
        # Clear the global pool for other tests
        pool1.clear_pool()

    def test_global_allocate_deallocate(self):
        """Testa as funções auxiliares de alocação/liberação globais."""
        # Clear global pool first
        global_pool = get_global_memory_pool(10)
        global_pool.clear_pool()
        
        # Allocate a tensor using global function
        tensor = allocate_tensor((10, 10), DataType.FLOAT32)
        self.assertEqual(tensor.shape, torch.Size((10, 10)))
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Deallocate back to global pool
        success = deallocate_tensor(tensor, (10, 10))
        self.assertTrue(success)


class TestMemoryEfficiency(unittest.TestCase):
    """Testes de eficiência de memória e desempenho."""

    def test_memory_reuse_reduction(self):
        """Testa se o pool reduz alocações de memória."""
        pool = MemoryPool(max_pool_size_mb=50)  # Larger pool for this test
        
        # Track allocations before
        initial_stats = pool.get_stats()
        
        # Perform multiple allocate-release cycles
        for i in range(100):
            tensor = pool.get_tensor((100, 100), DataType.FLOAT32)  # ~40KB each
            pool.release_tensor(tensor, (100, 100))
        
        final_stats = pool.get_stats()
        
        # Most allocations after the first should come from the pool
        total_requests = final_stats['pool_hits'] + final_stats['pool_misses']
        pool_hit_rate = final_stats['pool_hits'] / total_requests if total_requests > 0 else 0
        
        # We expect a high hit rate after the first few allocations
        self.assertGreater(pool_hit_rate, 0.8, f"Pool hit rate too low: {pool_hit_rate}")


def run_tests():
    """Executa todos os testes."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()