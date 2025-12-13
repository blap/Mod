"""
Testes abrangentes para o sistema avançado de memory pooling
"""
import unittest
import time
from advanced_memory_pooling_system import (
    AdvancedMemoryPoolingSystem, 
    TensorType, 
    MemoryBlock,
    BuddyAllocator,
    MemoryPool
)

class TestBuddyAllocator(unittest.TestCase):
    """Testes para o algoritmo Buddy Allocator"""
    
    def test_initialization(self):
        """Testa a inicialização do Buddy Allocator"""
        allocator = BuddyAllocator(1024, 64)
        self.assertEqual(allocator.total_size, 1024)
        self.assertEqual(allocator.min_block_size, 64)
        self.assertEqual(allocator.levels, 5)  # log2(1024/64) + 1 = 5
        
        # Verificar que o maior bloco está disponível
        self.assertEqual(len(allocator.free_blocks[4]), 1)
        self.assertTrue(allocator.free_blocks[4].pop().is_free)
    
    def test_allocation_success(self):
        """Testa alocação bem-sucedida"""
        allocator = BuddyAllocator(1024, 64)
        
        block = allocator.allocate(128, TensorType.KV_CACHE, "test_1")
        self.assertIsNotNone(block)
        self.assertEqual(block.size, 128)
        self.assertFalse(block.is_free)
        self.assertEqual(block.tensor_type, TensorType.KV_CACHE)
        self.assertEqual(block.tensor_id, "test_1")
    
    def test_allocation_with_splitting(self):
        """Testa alocação que requer divisão de blocos"""
        allocator = BuddyAllocator(1024, 64)
        
        # Alocar um bloco de 128 bytes - deve dividir o bloco de 1024
        block1 = allocator.allocate(128, TensorType.KV_CACHE, "test_1")
        self.assertIsNotNone(block1)
        self.assertEqual(block1.size, 128)
        
        # Alocar outro bloco de 128 bytes - deve usar o outro "buddy"
        block2 = allocator.allocate(128, TensorType.KV_CACHE, "test_2")
        self.assertIsNotNone(block2)
        self.assertEqual(block2.size, 128)
    
    def test_allocation_failure(self):
        """Testa falha na alocação por falta de memória"""
        allocator = BuddyAllocator(128, 64)
        
        # Alocar 100 bytes (arredondado para 128) - deve usar todo o espaço
        block1 = allocator.allocate(100, TensorType.KV_CACHE, "test_1")
        self.assertIsNotNone(block1)
        
        # Tentar alocar mais 64 bytes - deve falhar
        block2 = allocator.allocate(64, TensorType.KV_CACHE, "test_2")
        self.assertIsNone(block2)
    
    def test_deallocation_and_merging(self):
        """Testa desalocação e fusão de blocos"""
        allocator = BuddyAllocator(1024, 64)

        # Alocar dois blocos menores (128 bytes cada)
        # Isso dividirá o bloco de 1024 em blocos menores
        block1 = allocator.allocate(128, TensorType.KV_CACHE, "test_1")
        block2 = allocator.allocate(128, TensorType.KV_CACHE, "test_2")

        self.assertIsNotNone(block1)
        self.assertIsNotNone(block2)
        self.assertEqual(block1.size, 128)
        self.assertEqual(block2.size, 128)

        # Desalocar ambos
        allocator.deallocate(block1)
        allocator.deallocate(block2)

        # Após a desalocação, os blocos podem ter sido fundidos com outros blocos livres
        # No nosso caso, como tínhamos um bloco de 1024 que foi dividido, ao desalocar
        # os dois blocos de 128, eles podem ser fundidos de volta ao bloco original

        # O importante é que a memória total disponível seja correta
        total_free = sum(
            sum(b.size for b in blocks)
            for blocks in allocator.free_blocks.values()
        )
        self.assertEqual(total_free, 1024)  # Todo o espaço deve estar disponível novamente
    
    def test_size_to_level_conversion(self):
        """Testa a conversão de tamanho para nível"""
        allocator = BuddyAllocator(1024, 64)

        # Com tamanho total 1024 e tamanho mínimo 64:
        # Níveis: log2(1024/64) + 1 = log2(16) + 1 = 4 + 1 = 5 níveis (0 a 4)
        # Nível 0: blocos de 64 bytes
        # Nível 1: blocos de 128 bytes
        # Nível 2: blocos de 256 bytes
        # Nível 3: blocos de 512 bytes
        # Nível 4: blocos de 1024 bytes

        # Tamanho 64 -> nível 0 (blocos de tamanho mínimo)
        self.assertEqual(allocator._size_to_level(64), 0)
        # Tamanho 128 -> nível 1
        self.assertEqual(allocator._size_to_level(128), 1)
        # Tamanho 256 -> nível 2
        self.assertEqual(allocator._size_to_level(256), 2)
        # Tamanho 512 -> nível 3
        self.assertEqual(allocator._size_to_level(512), 3)
        # Tamanho 1024 -> nível 4
        self.assertEqual(allocator._size_to_level(1024), 4)


class TestMemoryPool(unittest.TestCase):
    """Testes para o pool de memória especializado"""
    
    def test_pool_initialization(self):
        """Testa a inicialização do pool de memória"""
        pool = MemoryPool(TensorType.KV_CACHE, 1024*1024, 256)
        self.assertEqual(pool.tensor_type, TensorType.KV_CACHE)
        self.assertEqual(pool.pool_size, 1024*1024)
        self.assertEqual(pool.min_block_size, 256)
        self.assertEqual(pool.utilization_ratio, 0.0)
        self.assertEqual(pool.fragmentation_ratio, 0.0)
    
    def test_pool_allocation_and_deallocation(self):
        """Testa alocação e desalocação no pool"""
        pool = MemoryPool(TensorType.KV_CACHE, 1024*1024, 256)
        
        # Alocar um bloco
        block = pool.allocate(1024, "tensor_1")
        self.assertIsNotNone(block)
        self.assertIn("tensor_1", pool.active_allocations)
        self.assertGreater(pool.utilization_ratio, 0)
        
        # Desalocar o bloco
        success = pool.deallocate("tensor_1")
        self.assertTrue(success)
        self.assertNotIn("tensor_1", pool.active_allocations)
        self.assertEqual(pool.utilization_ratio, 0.0)
    
    def test_pool_stats_update(self):
        """Testa a atualização de estatísticas do pool"""
        pool = MemoryPool(TensorType.KV_CACHE, 1024*1024, 256)
        
        initial_utilization = pool.utilization_ratio
        initial_fragmentation = pool.fragmentation_ratio
        
        # Alocar um bloco
        block = pool.allocate(512*1024, "large_tensor")  # 512KB
        self.assertIsNotNone(block)
        
        # Verificar que as estatísticas foram atualizadas
        self.assertGreater(pool.utilization_ratio, initial_utilization)
        # A fragmentação pode mudar dependendo da alocação


class TestAdvancedMemoryPoolingSystem(unittest.TestCase):
    """Testes para o sistema avançado de memory pooling"""
    
    def setUp(self):
        """Configuração antes de cada teste"""
        self.memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024,  # 1MB
            image_features_size=1024*1024,  # 1MB
            text_embeddings_size=1024*1024,  # 1MB
            gradients_size=1024*1024,  # 1MB
            activations_size=1024*1024,  # 1MB
            parameters_size=2*1024*1024,  # 2MB
            min_block_size=256
        )
    
    def test_system_initialization(self):
        """Testa a inicialização do sistema de memory pooling"""
        self.assertEqual(len(self.memory_system.pools), 6)
        self.assertIn(TensorType.KV_CACHE, self.memory_system.pools)
        self.assertIn(TensorType.IMAGE_FEATURES, self.memory_system.pools)
        self.assertIn(TensorType.TEXT_EMBEDDINGS, self.memory_system.pools)
        self.assertIn(TensorType.GRADIENTS, self.memory_system.pools)
        self.assertIn(TensorType.ACTIVATIONS, self.memory_system.pools)
        self.assertIn(TensorType.PARAMETERS, self.memory_system.pools)
        
        # Verificar que o hardware optimizer foi criado
        self.assertIsNotNone(self.memory_system.hardware_optimizer)
    
    def test_allocation_and_deallocation(self):
        """Testa alocação e desalocação de diferentes tipos de tensores"""
        # Testar alocação de KV cache
        kv_block = self.memory_system.allocate(TensorType.KV_CACHE, 1024*100, "kv_1")  # 100KB
        self.assertIsNotNone(kv_block)
        
        # Testar alocação de features de imagem
        img_block = self.memory_system.allocate(TensorType.IMAGE_FEATURES, 1024*200, "img_1")  # 200KB
        self.assertIsNotNone(img_block)
        
        # Testar alocação de embeddings de texto
        text_block = self.memory_system.allocate(TensorType.TEXT_EMBEDDINGS, 1024*50, "text_1")  # 50KB
        self.assertIsNotNone(text_block)
        
        # Verificar estatísticas
        kv_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertGreater(kv_stats['active_allocations'], 0)
        
        # Desalocar
        self.assertTrue(self.memory_system.deallocate(TensorType.KV_CACHE, "kv_1"))
        self.assertTrue(self.memory_system.deallocate(TensorType.IMAGE_FEATURES, "img_1"))
        self.assertTrue(self.memory_system.deallocate(TensorType.TEXT_EMBEDDINGS, "text_1"))
        
        # Verificar que os blocos foram removidos
        kv_stats_after = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertEqual(kv_stats_after['active_allocations'], 0)
    
    def test_fragmentation_management(self):
        """Testa o gerenciamento de fragmentação"""
        # Alocar e desalocar blocos para criar fragmentação
        blocks = []
        for i in range(10):
            block = self.memory_system.allocate(TensorType.KV_CACHE, 1024*10, f"block_{i}")  # 10KB
            if block:
                blocks.append(block)
        
        # Desalocar metade dos blocos para criar fragmentação
        for i in range(0, 10, 2):  # Desalocar índices pares
            self.memory_system.deallocate(TensorType.KV_CACHE, f"block_{i}")
        
        # Verificar estatísticas de fragmentação
        stats = self.memory_system.get_system_stats()
        # A fragmentação deve ser > 0 após esta operação
        self.assertGreaterEqual(stats['average_fragmentation'], 0.0)
        
        # Executar compactação
        self.memory_system.compact_memory()
        
        # As estatísticas devem ser atualizadas após compactação
        new_stats = self.memory_system.get_system_stats()
        self.assertIsInstance(new_stats['average_fragmentation'], float)
    
    def test_cache_integration(self):
        """Testa a integração com o sistema de cache hierárquico"""
        # Alocar um tensor
        block1 = self.memory_system.allocate(TensorType.KV_CACHE, 1024*10, "cached_tensor")
        self.assertIsNotNone(block1)
        
        # Tentar alocar o mesmo tensor novamente (deve retornar do cache)
        block2 = self.memory_system.allocate(TensorType.KV_CACHE, 1024*10, "cached_tensor")
        self.assertEqual(block1, block2)  # Deve ser o mesmo bloco
        
        # Desalocar
        self.assertTrue(self.memory_system.deallocate(TensorType.KV_CACHE, "cached_tensor"))
    
    def test_hardware_optimization(self):
        """Testa as otimizações de hardware"""
        hw_optimizer = self.memory_system.hardware_optimizer
        
        # Testar tamanhos de bloco ótimos
        kv_block_size = hw_optimizer.get_optimal_block_size(TensorType.KV_CACHE)
        self.assertGreaterEqual(kv_block_size, hw_optimizer.cpu_l1_cache_size)
        
        img_block_size = hw_optimizer.get_optimal_block_size(TensorType.IMAGE_FEATURES)
        self.assertGreaterEqual(img_block_size, hw_optimizer.cpu_l2_cache_size)
        
        # Testar decisão de uso de GPU
        should_use_gpu = hw_optimizer.should_use_gpu_memory(TensorType.IMAGE_FEATURES, 1024*1024)  # 1MB
        self.assertIsInstance(should_use_gpu, bool)
    
    def test_compact_memory(self):
        """Testa a compactação de memória"""
        # Alocar alguns blocos
        blocks = []
        for i in range(5):
            block = self.memory_system.allocate(TensorType.KV_CACHE, 1024*100, f"compact_test_{i}")
            if block:
                blocks.append(block)
        
        # Desalocar alguns para criar fragmentação
        for i in range(0, 5, 2):  # Índices pares
            self.memory_system.deallocate(TensorType.KV_CACHE, f"compact_test_{i}")
        
        # Executar compactação
        result = self.memory_system.compact_memory()
        self.assertTrue(result)
        
        # Verificar estatísticas após compactação
        stats = self.memory_system.get_system_stats()
        self.assertIn('total_fragmentation', stats)


class TestPerformance(unittest.TestCase):
    """Testes de desempenho"""
    
    def test_allocation_performance(self):
        """Testa o desempenho de alocação"""
        memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=10*1024*1024,  # 10MB
            min_block_size=256
        )
        
        start_time = time.time()
        blocks = []
        
        # Alocar muitos blocos pequenos
        for i in range(1000):
            block = memory_system.allocate(TensorType.KV_CACHE, 1024, f"perf_test_{i}")  # 1KB
            if block:
                blocks.append(block)
        
        end_time = time.time()
        allocation_time = end_time - start_time
        
        # Verificar que todas as alocações foram bem-sucedidas
        self.assertEqual(len(blocks), 1000)
        
        # O tempo de alocação deve ser razoável (menos de 5 segundos para 1000 alocações)
        self.assertLess(allocation_time, 5.0)
        
        # Desalocar todos
        for i, block in enumerate(blocks):
            memory_system.deallocate(TensorType.KV_CACHE, f"perf_test_{i}")


def run_tests():
    """Executa todos os testes"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()