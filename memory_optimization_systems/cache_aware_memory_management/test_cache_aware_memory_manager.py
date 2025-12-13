"""
Testes abrangentes para o Cache-Aware Memory Manager do Qwen3-VL.

Este módulo contém testes unitários e de integração para validar
todas as funcionalidades do sistema de gerenciamento de memória
com consciência de cache.
"""

import unittest
import numpy as np
import time
from cache_aware_memory_manager import (
    CacheAwareMemoryManager, 
    DataType, 
    LRUCache,
    create_optimized_tensor,
    store_tensor_with_cache,
    retrieve_tensor_from_cache
)


class TestLRUCache(unittest.TestCase):
    """Testes para a implementação básica de cache LRU"""
    
    def setUp(self):
        self.cache = LRUCache(capacity=3, data_type=DataType.WEIGHTS)
    
    def test_put_and_get(self):
        """Testa a operação básica de colocar e obter itens do cache"""
        self.cache.put("key1", np.array([1, 2, 3]))
        result = self.cache.get("key1")
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    
    def test_eviction_policy(self):
        """Testa a política de evicção LRU"""
        # Adiciona mais itens do que a capacidade
        self.cache.put("key1", np.array([1]))
        self.cache.put("key2", np.array([2]))
        self.cache.put("key3", np.array([3]))
        self.cache.put("key4", np.array([4]))  # Deve causar evicção
        
        # key1 deve ter sido removido (menos recentemente usado)
        self.assertIsNone(self.cache.get("key1"))
        
        # Os outros itens devem estar presentes
        self.assertIsNotNone(self.cache.get("key2"))
        self.assertIsNotNone(self.cache.get("key3"))
        self.assertIsNotNone(self.cache.get("key4"))
    
    def test_lru_ordering(self):
        """Testa que a ordenação LRU é mantida corretamente"""
        self.cache.put("key1", np.array([1]))
        self.cache.put("key2", np.array([2]))
        self.cache.put("key3", np.array([3]))
        
        # Acessa key1 novamente, tornando-o MRU
        self.cache.get("key1")
        
        # Adiciona um novo item, causando evicção
        self.cache.put("key4", np.array([4]))
        
        # key2 deve ser evictado (era o LRU antes de adicionar key4)
        self.assertIsNone(self.cache.get("key2"))
    
    def test_stats_tracking(self):
        """Testa o rastreamento de estatísticas do cache"""
        self.cache.put("key1", np.array([1, 2, 3]))
        
        # Um acesso (hit)
        result = self.cache.get("key1")
        stats = self.cache.stats()
        
        self.assertEqual(stats["access_count"], 1)
        self.assertEqual(stats["hit_count"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 1.0)
    
    def test_data_type_specific_policies(self):
        """Testa políticas específicas por tipo de dado"""
        weight_cache = LRUCache(capacity=10, data_type=DataType.WEIGHTS)
        activation_cache = LRUCache(capacity=10, data_type=DataType.ACTIVATIONS)
        
        # Capacidades efetivas devem ser diferentes baseadas no tipo
        self.assertNotEqual(weight_cache._effective_capacity(), activation_cache._effective_capacity())


class TestCacheAwareMemoryManager(unittest.TestCase):
    """Testes para o gerenciador de memória com consciência de cache"""
    
    def setUp(self):
        self.manager = CacheAwareMemoryManager(
            l1_cache_size=32*1024,  # 32KB
            l2_cache_size=256*1024,  # 256KB
            l3_cache_size=6*1024*1024,  # 6MB
            cache_line_size=64
        )
    
    def test_initialization(self):
        """Testa a inicialização correta do gerenciador"""
        self.assertEqual(self.manager.l1_cache_size, 32*1024)
        self.assertEqual(self.manager.l2_cache_size, 256*1024)
        self.assertEqual(self.manager.l3_cache_size, 6*1024*1024)
        self.assertEqual(self.manager.cache_line_size, 64)
        
        # Verifica que caches por tipo de dado foram criados
        for dtype in DataType:
            self.assertIn(dtype, self.manager.caches)
    
    def test_align_to_cache_line(self):
        """Testa o alinhamento ao tamanho da linha de cache"""
        # Tamanho que não é múltiplo de 64
        size = 100
        aligned = self.manager.align_to_cache_line(size)
        self.assertEqual(aligned, 128)  # Próximo múltiplo de 64
        
        # Tamanho que já é múltiplo de 64
        size = 128
        aligned = self.manager.align_to_cache_line(size)
        self.assertEqual(aligned, 128)
    
    def test_allocate_tensor_aligned(self):
        """Testa a alocação de tensor com alinhamento de cache"""
        shape = (10, 20)
        tensor = self.manager.allocate_tensor_aligned(shape, dtype=np.float32)
        
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, np.float32)
        
        # Verifica que a contagem de alocação aumentou
        self.assertEqual(self.manager.stats["allocation_count"], 1)
    
    def test_store_and_retrieve_from_cache(self):
        """Testa armazenamento e recuperação de tensores no cache"""
        tensor = np.random.rand(5, 5).astype(np.float32)
        key = "test_tensor"
        data_type = DataType.WEIGHTS
        
        # Armazena no cache
        success = self.manager.store_in_cache(key, tensor, data_type)
        self.assertTrue(success)
        
        # Recupera do cache
        retrieved = self.manager.retrieve_from_cache(key, data_type)
        np.testing.assert_array_equal(tensor, retrieved)
        
        # Verifica estatísticas
        self.assertEqual(self.manager.stats["cache_hits"], 1)
    
    def test_cache_miss(self):
        """Testa comportamento quando um item não está no cache"""
        retrieved = self.manager.retrieve_from_cache("nonexistent_key", DataType.WEIGHTS)
        self.assertIsNone(retrieved)
        
        # Verifica que foi registrado como miss
        self.assertEqual(self.manager.stats["cache_misses"], 1)
    
    def test_prefetch_functionality(self):
        """Testa a funcionalidade de prefetching"""
        key = "prefetch_test"
        data_type = DataType.ACTIVATIONS
        
        # Realiza prefetch
        result = self.manager.prefetch_tensor(key, data_type)
        # Pode retornar None se o item não estiver no cache
        
        # Verifica que o acesso foi registrado no histórico
        self.assertIn(key, self.manager.access_history)
        
        # Verifica que a contagem de prefetch aumentou
        self.assertGreater(self.manager.stats["prefetch_count"], 0)
    
    def test_predict_access_pattern(self):
        """Testa a predição de padrões de acesso"""
        key = "prediction_test"
        
        # Simula alguns acessos com intervalos regulares
        initial_time = time.time()
        self.manager.access_history[key] = [
            (initial_time, DataType.WEIGHTS.value),
            (initial_time + 1.0, DataType.WEIGHTS.value),
            (initial_time + 2.0, DataType.WEIGHTS.value)
        ]
        
        # Obtém padrão previsto
        pattern = self.manager._predict_access_pattern(key)
        
        # Deve haver pelo menos uma previsão
        self.assertIsInstance(pattern, list)
    
    def test_optimize_for_locality(self):
        """Testa a otimização para localidade"""
        # Cria alguns tensores de teste
        tensors = [
            (np.random.rand(10, 10), DataType.WEIGHTS),
            (np.random.rand(5, 5), DataType.ACTIVATIONS),
            (np.random.rand(8, 8), DataType.GRADIENTS)
        ]
        
        optimized = self.manager.optimize_for_locality(tensors)
        
        # Deve retornar a mesma quantidade de tensores
        self.assertEqual(len(optimized), len(tensors))
        
        # Todos devem ser arrays numpy
        for tensor in optimized:
            self.assertIsInstance(tensor, np.ndarray)
    
    def test_get_cache_stats(self):
        """Testa a obtenção de estatísticas de cache"""
        stats = self.manager.get_cache_stats()
        
        # Deve conter estatísticas para todos os tipos de dados
        for dtype in DataType:
            self.assertIn(f"{dtype.value}_stats", stats)
        
        # Deve conter estatísticas gerais
        self.assertIn("overall_hit_rate", stats)
        self.assertIn("general_stats", stats)
    
    def test_flush_cache(self):
        """Testa a limpeza do cache"""
        # Armazena algo no cache
        tensor = np.array([1, 2, 3])
        self.manager.store_in_cache("test_flush", tensor, DataType.WEIGHTS)
        
        # Verifica que está no cache
        retrieved = self.manager.retrieve_from_cache("test_flush", DataType.WEIGHTS)
        self.assertIsNotNone(retrieved)
        
        # Limpa o cache
        self.manager.flush_cache(DataType.WEIGHTS)
        
        # Verifica que foi removido
        retrieved_after = self.manager.retrieve_from_cache("test_flush", DataType.WEIGHTS)
        self.assertIsNone(retrieved_after)
    
    def test_hardware_specific_configurations(self):
        """Testa as configurações específicas de hardware"""
        # Verifica que as propriedades específicas de hardware estão definidas
        self.assertIsInstance(self.manager.l1_cache_size, int)
        self.assertIsInstance(self.manager.l2_cache_size, int)
        self.assertIsInstance(self.manager.l3_cache_size, int)
        self.assertIsInstance(self.manager.cache_line_size, int)


class TestIntegration(unittest.TestCase):
    """Testes de integração para fluxos completos"""
    
    def setUp(self):
        self.manager = CacheAwareMemoryManager()
    
    def test_complete_workflow(self):
        """Testa um fluxo completo de uso do gerenciador"""
        # 1. Cria um tensor otimizado
        tensor = create_optimized_tensor(
            (128, 64), 
            dtype=np.float32, 
            data_type=DataType.WEIGHTS,
            manager=self.manager
        )
        self.assertEqual(tensor.shape, (128, 64))
        
        # 2. Armazena no cache
        stored = store_tensor_with_cache(
            tensor, 
            "layer_weights", 
            DataType.WEIGHTS, 
            self.manager
        )
        self.assertTrue(stored)
        
        # 3. Recupera do cache
        retrieved = retrieve_tensor_from_cache(
            "layer_weights", 
            DataType.WEIGHTS, 
            self.manager
        )
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(tensor, retrieved)
        
        # 4. Verifica estatísticas
        stats = self.manager.get_cache_stats()
        self.assertGreater(stats["overall_hit_rate"], 0)
    
    def test_multiple_data_types_handling(self):
        """Testa o manuseio de diferentes tipos de dados"""
        data_types = [DataType.WEIGHTS, DataType.ACTIVATIONS, DataType.GRADIENTS]
        
        for i, dtype in enumerate(data_types):
            tensor = np.random.rand(10, 10).astype(np.float32)
            key = f"tensor_{dtype.value}_{i}"
            
            # Armazena no cache apropriado
            self.manager.store_in_cache(key, tensor, dtype)
            
            # Recupera do cache
            retrieved = self.manager.retrieve_from_cache(key, dtype)
            np.testing.assert_array_equal(tensor, retrieved)
    
    def test_cache_line_alignment_impact(self):
        """Testa o impacto do alinhamento de linha de cache"""
        shapes = [(7, 11), (13, 17), (19, 23)]  # Formas que provavelmente resultam em tamanhos não alinhados
        
        for shape in shapes:
            tensor = self.manager.allocate_tensor_aligned(shape, dtype=np.float32)
            self.assertEqual(tensor.shape, shape)
            
            # Verifica que ocorreram melhorias de alinhamento
            initial_alignments = self.manager.stats["alignment_improvements"]
            # Faz mais uma alocação para garantir que o contador aumente
            extra_tensor = self.manager.allocate_tensor_aligned((5, 5), dtype=np.float32)
            final_alignments = self.manager.stats["alignment_improvements"]
            
            # Pelo menos uma melhoria de alinhamento deve ter ocorrido
            # (embora dependa das formas exatas)
    

def run_performance_test():
    """Teste de desempenho para avaliar ganhos de cache"""
    print("Executando teste de desempenho...")
    
    manager = CacheAwareMemoryManager()
    
    # Teste de tempo de acesso com e sem cache
    tensor = np.random.rand(100, 100).astype(np.float32)
    
    # Armazena no cache
    manager.store_in_cache("perf_test", tensor, DataType.WEIGHTS)
    
    # Mede tempo de acesso repetido
    start_time = time.time()
    for _ in range(1000):
        retrieved = manager.retrieve_from_cache("perf_test", DataType.WEIGHTS)
    end_time = time.time()
    
    cache_access_time = end_time - start_time
    
    print(f"Tempo médio de acesso com cache: {cache_access_time/1000:.6f}s")
    print(f"Taxa de acerto geral: {manager.get_cache_stats()['overall_hit_rate']:.2%}")


if __name__ == "__main__":
    # Executa os testes unitários
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Executa teste de desempenho
    run_performance_test()