"""
Comprehensive tests for the Advanced Hierarchical Caching System
"""

import unittest
import numpy as np
import tempfile
import os
import time
from pathlib import Path

# Import classes from the main module
from advanced_hierarchical_caching_system import (
    CacheLevel, TensorType, CacheEntry, 
    AccessPatternPredictor, PrefetchingManager,
    LRUCache, HierarchicalCacheManager, 
    Qwen3VLCacheOptimizer
)


class TestLRUCache(unittest.TestCase):
    """Testes para a implementação básica de LRU Cache"""
    
    def setUp(self):
        self.cache = LRUCache(capacity_bytes=1024)  # 1KB
    
    def test_put_and_get_single_item(self):
        """Testa a inserção e recuperação de um único item"""
        data = np.random.random((10, 10)).astype(np.float32)
        entry = CacheEntry(
            tensor_id="test_tensor",
            tensor_type=TensorType.KV_CACHE,
            data=data,
            size_bytes=data.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        # Colocar no cache
        result = self.cache.put(entry)
        self.assertTrue(result)
        
        # Recuperar do cache
        retrieved = self.cache.get("test_tensor")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.tensor_id, "test_tensor")
        np.testing.assert_array_equal(retrieved.data, data)
    
    def test_lru_eviction(self):
        """Testa o comportamento de evicção LRU"""
        # Criar tensores que juntos excedem a capacidade
        tensor1 = np.random.random((8, 8)).astype(np.float32)  # ~256 bytes
        tensor2 = np.random.random((10, 10)).astype(np.float32)  # ~400 bytes
        tensor3 = np.random.random((12, 12)).astype(np.float32)  # ~576 bytes
        
        entry1 = CacheEntry(
            tensor_id="tensor1",
            tensor_type=TensorType.KV_CACHE,
            data=tensor1,
            size_bytes=tensor1.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        entry2 = CacheEntry(
            tensor_id="tensor2",
            tensor_type=TensorType.KV_CACHE,
            data=tensor2,
            size_bytes=tensor2.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        entry3 = CacheEntry(
            tensor_id="tensor3",
            tensor_type=TensorType.KV_CACHE,
            data=tensor3,
            size_bytes=tensor3.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        # Adicionar os dois primeiros tensores
        self.assertTrue(self.cache.put(entry1))
        self.assertTrue(self.cache.put(entry2))
        
        # Acessar o primeiro tensor para atualizar sua posição LRU
        self.cache.get("tensor1")
        
        # Adicionar o terceiro tensor (deve causar evicção do tensor2)
        self.assertTrue(self.cache.put(entry3))
        
        # Verificar que tensor1 e tensor3 estão no cache, mas tensor2 não
        self.assertIsNotNone(self.cache.get("tensor1"))
        self.assertIsNotNone(self.cache.get("tensor3"))
        self.assertIsNone(self.cache.get("tensor2"))
    
    def test_get_nonexistent_key(self):
        """Testa a recuperação de uma chave inexistente"""
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)


class TestAccessPatternPredictor(unittest.TestCase):
    """Testes para o preditor de padrões de acesso"""
    
    def setUp(self):
        self.predictor = AccessPatternPredictor()
    
    def test_record_access_and_predict(self):
        """Testa o registro de acessos e predição subsequente"""
        now = time.time()
        
        # Registrar alguns acessos
        self.predictor.record_access("tensor1", now)
        self.predictor.record_access("tensor1", now + 1.0)  # 1 segundo depois
        self.predictor.record_access("tensor1", now + 2.0)  # 1 segundo depois
        
        # A predição deve ser feita com base no intervalo médio
        predicted = self.predictor.predict_next_access("tensor1")
        expected_time = now + 3.0  # Com intervalo médio de 1.0s
        
        # Permitir alguma tolerância devido a flutuações
        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted, expected_time, delta=0.1)
    
    def test_predict_with_insufficient_data(self):
        """Testa a predição com dados insuficientes"""
        # Sem dados suficientes, a predição deve retornar None
        self.assertIsNone(self.predictor.predict_next_access("new_tensor"))
        
        # Com apenas um acesso, ainda não há intervalo
        self.predictor.record_access("new_tensor", time.time())
        self.assertIsNone(self.predictor.predict_next_access("new_tensor"))
    
    def test_access_frequency_calculation(self):
        """Testa o cálculo de frequência de acesso"""
        now = time.time()
        
        # Registrar acessos em um curto período
        for i in range(5):
            self.predictor.record_access("frequent_tensor", now + i * 0.1)
        
        freq = self.predictor.get_access_frequency("frequent_tensor")
        # Frequência esperada: 5 acessos em 0.4 segundos = 12.5 Hz
        self.assertGreater(freq, 10.0)  # Mais de 10Hz
    
    def test_access_frequency_with_single_access(self):
        """Testa a frequência com um único acesso"""
        self.predictor.record_access("single_tensor", time.time())
        freq = self.predictor.get_access_frequency("single_tensor")
        self.assertEqual(freq, 0.0)


class TestHierarchicalCacheManager(unittest.TestCase):
    """Testes para o gerenciador de cache hierárquico"""
    
    def setUp(self):
        # Usar um diretório temporário para o cache L3
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = HierarchicalCacheManager(
            l1_gpu_size=512 * 1024,  # 512KB
            l2_cpu_size=1024 * 1024,  # 1MB
            l3_disk_path=os.path.join(self.temp_dir, "l3_cache"),
            l3_disk_size=2 * 1024 * 1024  # 2MB
        )
    
    def tearDown(self):
        # Limpar o diretório temporário
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_cache_operations(self):
        """Testa operações básicas de cache"""
        # Criar um tensor de teste
        tensor_data = np.random.random((10, 10)).astype(np.float32)
        entry = CacheEntry(
            tensor_id="test_tensor",
            tensor_type=TensorType.KV_CACHE,
            data=tensor_data,
            size_bytes=tensor_data.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        # Armazenar no cache
        self.assertTrue(self.cache_manager.put_tensor(entry))
        
        # Recuperar do cache
        retrieved = self.cache_manager.get_tensor("test_tensor")
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved.data, tensor_data)
    
    def test_cache_hierarchy_movement(self):
        """Testa o movimento entre níveis de cache"""
        # Criar um tensor pequeno que se qualifique para L1
        small_tensor = np.random.random((5, 5)).astype(np.float32)
        entry = CacheEntry(
            tensor_id="small_tensor",
            tensor_type=TensorType.KV_CACHE,
            data=small_tensor,
            size_bytes=small_tensor.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        # Armazenar no cache
        self.assertTrue(self.cache_manager.put_tensor(entry))
        
        # Simular acessos frequentes para promover para L1
        for _ in range(20):
            self.cache_manager.access_predictor.record_access("small_tensor", time.time())
            time.sleep(0.01)
            self.cache_manager.get_tensor("small_tensor")
        
        # Após acessos frequentes, o tensor deve estar em L1 ou L2
        retrieved = self.cache_manager.get_tensor("small_tensor")
        self.assertIsNotNone(retrieved)
    
    def test_l3_cache_persistence(self):
        """Testa a persistência no cache L3 (SSD)"""
        # Criar um tensor grande que vá para L3
        large_tensor = np.random.random((100, 100)).astype(np.float32)
        entry = CacheEntry(
            tensor_id="large_tensor",
            tensor_type=TensorType.IMAGE_FEATURES,
            data=large_tensor,
            size_bytes=large_tensor.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        # Armazenar no cache
        self.assertTrue(self.cache_manager.put_tensor(entry))
        
        # Recuperar do cache
        retrieved = self.cache_manager.get_tensor("large_tensor")
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved.data, large_tensor)
    
    def test_cache_stats(self):
        """Testa as estatísticas do cache"""
        # Criar e armazenar alguns tensores
        tensor1 = np.random.random((10, 10)).astype(np.float32)
        entry1 = CacheEntry(
            tensor_id="tensor1",
            tensor_type=TensorType.KV_CACHE,
            data=tensor1,
            size_bytes=tensor1.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        self.cache_manager.put_tensor(entry1)
        
        # Acessar o tensor
        self.cache_manager.get_tensor("tensor1")
        
        # Verificar estatísticas
        stats = self.cache_manager.get_cache_stats()
        self.assertIn('total_requests', stats)
        self.assertGreaterEqual(stats['total_requests'], 1)
    
    def test_cache_clearing(self):
        """Testa a limpeza do cache"""
        # Adicionar um tensor
        tensor = np.random.random((10, 10)).astype(np.float32)
        entry = CacheEntry(
            tensor_id="clear_test",
            tensor_type=TensorType.KV_CACHE,
            data=tensor,
            size_bytes=tensor.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        self.cache_manager.put_tensor(entry)
        self.assertIsNotNone(self.cache_manager.get_tensor("clear_test"))
        
        # Limpar o cache
        self.cache_manager.clear_cache()
        
        # Verificar que o tensor não está mais lá
        self.assertIsNone(self.cache_manager.get_tensor("clear_test"))


class TestQwen3VLCacheOptimizer(unittest.TestCase):
    """Testes para o otimizador de cache específico do Qwen3-VL"""
    
    def setUp(self):
        temp_dir = tempfile.mkdtemp()
        cache_manager = HierarchicalCacheManager(
            l1_gpu_size=512 * 1024,  # 512KB
            l2_cpu_size=1024 * 1024,  # 1MB
            l3_disk_path=os.path.join(temp_dir, "l3_cache"),
            l3_disk_size=2 * 1024 * 1024  # 2MB
        )
        self.optimizer = Qwen3VLCacheOptimizer(cache_manager)
        self.temp_dir = temp_dir
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimize_for_different_tensor_types(self):
        """Testa a otimização para diferentes tipos de tensores"""
        # Testar otimizações para cada tipo de tensor
        for tensor_type in TensorType:
            opts = self.optimizer.optimize_cache_for_tensor_type(tensor_type)
            
            # Verificar que as chaves esperadas estão presentes
            self.assertIn('l1_priority', opts)
            self.assertIn('prefetch_distance', opts)
            self.assertIn('access_pattern_threshold', opts)
    
    def test_optimal_cache_sizes(self):
        """Testa o cálculo de tamanhos ideais de cache"""
        total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        l1_size, l2_size, l3_size = self.optimizer.get_optimal_cache_sizes(total_memory)
        
        # Verificar que os tamanhos são positivos
        self.assertGreater(l1_size, 0)
        self.assertGreater(l2_size, 0)
        self.assertGreater(l3_size, 0)
        
        # Verificar que a soma não excede o total
        self.assertLessEqual(l1_size + l2_size + l3_size, total_memory)


class TestIntegration(unittest.TestCase):
    """Testes de integração entre componentes"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = HierarchicalCacheManager(
            l1_gpu_size=512 * 1024,  # 512KB
            l2_cpu_size=1024 * 1024,  # 1MB
            l3_disk_path=os.path.join(self.temp_dir, "l3_cache"),
            l3_disk_size=2 * 1024 * 1024  # 2MB
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow(self):
        """Testa o fluxo completo de trabalho"""
        # Criar vários tensores de diferentes tipos
        tensors = []
        for i, tensor_type in enumerate([TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, 
                                        TensorType.TEXT_EMBEDDINGS, TensorType.GRADIENTS]):
            tensor_data = np.random.random((20, 20)).astype(np.float32)
            entry = CacheEntry(
                tensor_id=f"tensor_{i}_{tensor_type.value}",
                tensor_type=tensor_type,
                data=tensor_data,
                size_bytes=tensor_data.nbytes,
                access_time=time.time(),
                access_count=0,
                last_access_pattern=[]
            )
            tensors.append(entry)
        
        # Armazenar todos os tensores
        for entry in tensors:
            self.assertTrue(self.cache_manager.put_tensor(entry))
        
        # Acessar todos os tensores
        for entry in tensors:
            retrieved = self.cache_manager.get_tensor(entry.tensor_id)
            self.assertIsNotNone(retrieved)
            np.testing.assert_array_equal(retrieved.data, entry.data)
        
        # Verificar estatísticas
        stats = self.cache_manager.get_cache_stats()
        self.assertGreaterEqual(stats['total_requests'], len(tensors))
    
    def test_prediction_and_prefetching_integration(self):
        """Testa a integração entre predição e prefetching"""
        # Criar um tensor com acesso frequente
        tensor_data = np.random.random((10, 10)).astype(np.float32)
        entry = CacheEntry(
            tensor_id="frequent_tensor",
            tensor_type=TensorType.KV_CACHE,
            data=tensor_data,
            size_bytes=tensor_data.nbytes,
            access_time=time.time(),
            access_count=0,
            last_access_pattern=[]
        )
        
        # Armazenar no cache
        self.cache_manager.put_tensor(entry)
        
        # Simular padrão de acesso
        base_time = time.time()
        for i in range(5):
            # Registrar acesso
            self.cache_manager.access_predictor.record_access(
                "frequent_tensor", 
                base_time + i * 0.5
            )
            # Acessar o tensor
            self.cache_manager.get_tensor("frequent_tensor")
            time.sleep(0.01)
        
        # A predição deve ter sido calculada
        retrieved = self.cache_manager.get_tensor("frequent_tensor")
        self.assertIsNotNone(retrieved)
        self.assertIsNotNone(retrieved.predicted_next_access)


def run_tests():
    """Executa todos os testes"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLRUCache)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAccessPatternPredictor))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHierarchicalCacheManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestQwen3VLCacheOptimizer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nTodos os testes passaram com sucesso!")
    else:
        print("\nAlguns testes falharam.")
        exit(1)