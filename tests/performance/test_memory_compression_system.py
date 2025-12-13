import numpy as np
import pytest
import torch
import os
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import threading

# Importar as classes que vamos implementar
from memory_compression_system import (
    QuantizationCompressor,
    SVDCompressor,
    SparseCompressor,
    AutoCompressor,
    MemoryCompressionManager,
    CompressionStats
)


class TestQuantizationCompressor:
    """Testes para o compressor de quantização INT8/FP16"""
    
    def test_int8_quantization_basic(self):
        """Testa quantização INT8 básica"""
        compressor = QuantizationCompressor()
        
        # Criar tensor de exemplo
        original_tensor = torch.randn(10, 10, dtype=torch.float32)
        
        # Comprimir
        compressed_data = compressor.compress(original_tensor, method='int8')
        
        # Verificar estrutura dos dados comprimidos
        assert 'method' in compressed_data
        assert 'quantized' in compressed_data
        assert 'scale' in compressed_data
        assert 'original_shape' in compressed_data
        assert compressed_data['method'] == 'int8'
        assert compressed_data['quantized'].dtype == torch.int8
        assert compressed_data['quantized'].shape == original_tensor.shape
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que o tensor foi reconstruído corretamente (com alguma perda de precisão)
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
        
        # Verificar que a diferença está dentro de limites razoáveis
        mse = torch.mean((original_tensor - decompressed_tensor) ** 2)
        assert mse.item() < 0.01  # Erro quadrático médio aceitável
    
    def test_fp16_quantization_basic(self):
        """Testa quantização FP16 básica"""
        compressor = QuantizationCompressor()
        
        # Criar tensor de exemplo
        original_tensor = torch.randn(10, 10, dtype=torch.float32)
        
        # Comprimir
        compressed_data = compressor.compress(original_tensor, method='fp16')
        
        # Verificar estrutura dos dados comprimidos
        assert 'method' in compressed_data
        assert 'quantized' in compressed_data
        assert 'original_shape' in compressed_data
        assert compressed_data['method'] == 'fp16'
        assert compressed_data['quantized'].dtype == torch.float16
        assert compressed_data['quantized'].shape == original_tensor.shape
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que o tensor foi reconstruído corretamente
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
        
        # Verificar que a diferença está dentro de limites razoáveis
        mse = torch.mean((original_tensor - decompressed_tensor) ** 2)
        assert mse.item() < 0.001  # Erro quadrático médio aceitável para FP16
    
    def test_int8_quantization_different_shapes(self):
        """Testa quantização INT8 com diferentes formatos de tensor"""
        compressor = QuantizationCompressor()
        
        shapes = [(100,), (50, 60), (10, 20, 30), (5, 4, 3, 2)]
        
        for shape in shapes:
            original_tensor = torch.randn(shape, dtype=torch.float32)
            
            # Comprimir
            compressed_data = compressor.compress(original_tensor, method='int8')
            
            # Descomprimir
            decompressed_tensor = compressor.decompress(compressed_data)
            
            # Verificar que as formas correspondem
            assert decompressed_tensor.shape == original_tensor.shape
            assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_fp16_quantization_different_shapes(self):
        """Testa quantização FP16 com diferentes formatos de tensor"""
        compressor = QuantizationCompressor()
        
        shapes = [(100,), (50, 60), (10, 20, 30), (5, 4, 3, 2)]
        
        for shape in shapes:
            original_tensor = torch.randn(shape, dtype=torch.float32)
            
            # Comprimir
            compressed_data = compressor.compress(original_tensor, method='fp16')
            
            # Descomprimir
            decompressed_tensor = compressor.decompress(compressed_data)
            
            # Verificar que as formas correspondem
            assert decompressed_tensor.shape == original_tensor.shape
            assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_quantization_invalid_method(self):
        """Testa quantização com método inválido"""
        compressor = QuantizationCompressor()
        
        original_tensor = torch.randn(10, 10, dtype=torch.float32)
        
        with pytest.raises(ValueError):
            compressor.compress(original_tensor, method='invalid_method')
    
    def test_quantization_preserves_zero_values(self):
        """Testa que a quantização preserva valores zero em tensores esparsos"""
        compressor = QuantizationCompressor()
        
        # Criar tensor com muitos zeros
        original_tensor = torch.zeros(20, 20, dtype=torch.float32)
        original_tensor[::2, ::2] = torch.randn(10, 10)  # Preencher apenas metade
        
        # Testar INT8
        compressed_data = compressor.compress(original_tensor, method='int8')
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que zeros são preservados aproximadamente
        original_zeros = (original_tensor == 0)
        decompressed_zeros = (decompressed_tensor == 0)
        
        # A maioria dos zeros deve ser preservada (pode haver pequenas diferenças devido à quantização)
        zero_preservation_rate = (original_zeros == decompressed_zeros).float().mean()
        assert zero_preservation_rate > 0.8  # Pelo menos 80% dos zeros preservados
    
    def test_dynamic_scaling(self):
        """Testa escalas dinâmicas na quantização"""
        compressor = QuantizationCompressor()
        
        # Testar com diferentes faixas de valores
        ranges = [(-1.0, 1.0), (-10.0, 10.0), (0.0, 100.0)]
        
        for min_val, max_val in ranges:
            original_tensor = torch.empty(20, 20).uniform_(min_val, max_val)
            
            # Comprimir
            compressed_data = compressor.compress(original_tensor, method='int8')
            
            # Verificar que a escala é apropriada para o intervalo
            assert compressed_data['scale'] > 0
            
            # Descomprimir
            decompressed_tensor = compressor.decompress(compressed_data)
            
            # Verificar que a forma e tipo são preservados
            assert decompressed_tensor.shape == original_tensor.shape
            assert decompressed_tensor.dtype == original_tensor.dtype


class TestSVDCompressor:
    """Testes para o compressor SVD"""
    
    def test_svd_compression_basic(self):
        """Testa compressão SVD básica"""
        compressor = SVDCompressor()
        
        # Criar tensor 2D para SVD
        original_tensor = torch.randn(50, 30, dtype=torch.float32)
        
        # Comprimir com rank adaptativo
        compressed_data = compressor.compress(original_tensor, adaptive_rank=True)
        
        # Verificar estrutura dos dados comprimidos
        assert 'method' in compressed_data
        assert 'u' in compressed_data
        assert 's' in compressed_data
        assert 'v' in compressed_data
        assert 'original_shape' in compressed_data
        assert compressed_data['method'] == 'svd'
        assert compressed_data['u'].shape[0] == original_tensor.shape[0]
        # A dimensão do rank truncado pode ser diferente da dimensão original
        assert compressed_data['u'].shape[1] == compressed_data['v'].shape[1]  # Mesmo rank para U e V
        assert compressed_data['v'].shape[0] == original_tensor.shape[1]  # Primeira dim de V é a segunda dim original
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_svd_compression_with_fixed_rank(self):
        """Testa compressão SVD com rank fixo"""
        compressor = SVDCompressor()
        
        # Criar tensor 2D
        original_tensor = torch.randn(40, 25, dtype=torch.float32)
        target_rank = 10
        
        # Comprimir com rank fixo
        compressed_data = compressor.compress(original_tensor, rank=target_rank)
        
        # Verificar que os componentes têm o rank apropriado
        assert compressed_data['u'].shape[1] == target_rank
        assert compressed_data['s'].shape[0] == target_rank
        assert compressed_data['v'].shape[1] == target_rank  # V tem formato (original_cols, rank)
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
    
    def test_svd_adaptive_rank_selection(self):
        """Testa seleção adaptativa de rank no SVD"""
        compressor = SVDCompressor()
        
        # Criar tensor com estrutura de baixo rank
        low_rank_tensor = torch.randn(30, 20, dtype=torch.float32)
        # Criar tensor de baixo rank multiplicando duas matrizes menores
        u_small = torch.randn(30, 5, dtype=torch.float32)
        v_small = torch.randn(5, 20, dtype=torch.float32)
        original_tensor = torch.mm(u_small, v_small)
        
        # Comprimir com rank adaptativo
        compressed_data = compressor.compress(original_tensor, adaptive_rank=True, compression_ratio=0.5)
        
        # O rank adaptativo deve ser menor que o mínimo entre as dimensões
        assert compressed_data['s'].shape[0] <= min(original_tensor.shape)
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
    
    def test_svd_compression_3d_tensor(self):
        """Testa compressão SVD com tensor 3D (deve falhar ou ser manipulado adequadamente)"""
        compressor = SVDCompressor()
        
        # Criar tensor 3D
        original_tensor = torch.randn(10, 20, 30, dtype=torch.float32)
        
        # A compressão SVD deve lidar com tensores 3D de alguma forma
        # Por exemplo, aplicando SVD a cada fatia
        try:
            compressed_data = compressor.compress(original_tensor, adaptive_rank=True)
            # Se não falhar, verificar a estrutura
            assert 'method' in compressed_data
            assert compressed_data['method'] == 'svd'
        except Exception as e:
            # Se falhar, deve ser uma exceção esperada
            assert "SVD" in str(e) or "2D" in str(e) or "dimension" in str(e).lower()
    
    def test_svd_compression_preserves_properties(self):
        """Testa que a compressão SVD preserva propriedades importantes"""
        compressor = SVDCompressor()
        
        # Criar tensor específico
        original_tensor = torch.tensor([[3.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
        
        # Comprimir
        compressed_data = compressor.compress(original_tensor, adaptive_rank=True)
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype


class TestSparseCompressor:
    """Testes para o compressor esparsidade"""
    
    def test_sparse_compression_basic(self):
        """Testa compressão esparsidade básica"""
        compressor = SparseCompressor()
        
        # Criar tensor esparsa (muitos zeros)
        original_tensor = torch.zeros(20, 20, dtype=torch.float32)
        original_tensor[::3, ::3] = torch.randn(7, 7)  # Preencher alguns elementos
        
        # Comprimir
        compressed_data = compressor.compress(original_tensor, sparsity_threshold=0.5)
        
        # Verificar estrutura dos dados comprimidos
        assert 'method' in compressed_data
        assert 'values' in compressed_data
        assert 'indices' in compressed_data
        assert 'original_shape' in compressed_data
        assert compressed_data['method'] == 'sparse'
        
        # Verificar que a compressão reduziu o número de elementos
        original_elements = original_tensor.numel()
        compressed_elements = compressed_data['values'].numel()
        assert compressed_elements < original_elements
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
        
        # Verificar que os valores não-zero correspondem
        non_zero_original = original_tensor != 0
        non_zero_decompressed = decompressed_tensor != 0
        assert torch.equal(non_zero_original, non_zero_decompressed)
        assert torch.allclose(
            original_tensor[non_zero_original],
            decompressed_tensor[non_zero_decompressed],
            atol=1e-6
        )
    
    def test_sparse_compression_different_thresholds(self):
        """Testa compressão esparsidade com diferentes thresholds"""
        compressor = SparseCompressor()
        
        # Criar tensor com diferentes níveis de esparsidade
        original_tensor = torch.zeros(30, 30, dtype=torch.float32)
        original_tensor[:10, :10] = torch.randn(10, 10)  # 1/9 é não-zero
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            compressed_data = compressor.compress(original_tensor, sparsity_threshold=threshold)
            
            # Verificar que a compressão foi aplicada corretamente
            assert 'method' in compressed_data
            assert compressed_data['method'] == 'sparse'
    
    def test_sparse_compression_3d_tensor(self):
        """Testa compressão esparsidade com tensor 3D"""
        compressor = SparseCompressor()
        
        # Criar tensor 3D esparsa
        original_tensor = torch.zeros(5, 10, 8, dtype=torch.float32)
        original_tensor[::2, ::3, ::2] = torch.randn(3, 4, 4)  # Preencher alguns elementos
        
        # Comprimir
        compressed_data = compressor.compress(original_tensor, sparsity_threshold=0.5)
        
        # Verificar estrutura
        assert 'method' in compressed_data
        assert compressed_data['method'] == 'sparse'
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_sparse_compression_full_tensor(self):
        """Testa compressão esparsidade com tensor sem zeros (não deve comprimir bem)"""
        compressor = SparseCompressor()
        
        # Criar tensor denso (sem zeros)
        original_tensor = torch.randn(10, 10, dtype=torch.float32)
        
        # Comprimir com threshold alto (não deve ser eficaz)
        compressed_data = compressor.compress(original_tensor, sparsity_threshold=0.9)
        
        # O tensor comprimido deve ter todos os valores (nenhuma compressão efetiva)
        # ou o compressor deve detectar que não é eficiente e não comprimir
        assert 'method' in compressed_data
        assert compressed_data['method'] in ['sparse', 'none']
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype


class TestAutoCompressor:
    """Testes para o compressor automático"""
    
    def test_auto_compression_selection_quantized_tensor(self):
        """Testa seleção automática para tensor com baixa variância (deve escolher quantização)"""
        compressor = AutoCompressor()
        
        # Criar tensor com baixa variância
        original_tensor = torch.ones(20, 20, dtype=torch.float32) * 0.5 + torch.randn(20, 20) * 0.01
        
        # Comprimir automaticamente
        compressed_data = compressor.compress(original_tensor)
        
        # O compressor automático deve escolher o método mais apropriado
        # Para este caso, provavelmente quantização
        assert 'method' in compressed_data
        assert compressed_data['method'] in ['int8', 'fp16', 'svd', 'sparse']
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_auto_compression_selection_sparse_tensor(self):
        """Testa seleção automática para tensor esparsa (deve escolher esparsidade)"""
        compressor = AutoCompressor()
        
        # Criar tensor esparsa
        original_tensor = torch.zeros(20, 20, dtype=torch.float32)
        original_tensor[::4, ::4] = torch.randn(5, 5)  # Poucos elementos não-zero
        
        # Comprimir automaticamente
        compressed_data = compressor.compress(original_tensor)
        
        # O compressor automático deve detectar a esparsidade e escolher o método apropriado
        assert 'method' in compressed_data
        # Pode ser 'sparse' ou outro método dependendo da implementação
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_auto_compression_selection_low_rank_tensor(self):
        """Testa seleção automática para tensor de baixo rank (deve escolher SVD)"""
        compressor = AutoCompressor()
        
        # Criar tensor de baixo rank
        u = torch.randn(30, 5, dtype=torch.float32)
        v = torch.randn(5, 20, dtype=torch.float32)
        original_tensor = torch.mm(u, v)
        
        # Comprimir automaticamente
        compressed_data = compressor.compress(original_tensor)
        
        # O compressor automático deve detectar o potencial para compressão SVD
        assert 'method' in compressed_data
        
        # Descomprimir
        decompressed_tensor = compressor.decompress(compressed_data)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_auto_compression_preserves_properties(self):
        """Testa que a compressão automática preserva propriedades importantes"""
        compressor = AutoCompressor()
        
        # Testar com diferentes tipos de tensores
        test_cases = [
            torch.randn(15, 15, dtype=torch.float32),  # Tensor aleatório
            torch.zeros(15, 15, dtype=torch.float32),   # Tensor de zeros
            torch.ones(15, 15, dtype=torch.float32),    # Tensor de uns
        ]
        
        for original_tensor in test_cases:
            compressed_data = compressor.compress(original_tensor)
            decompressed_tensor = compressor.decompress(compressed_data)
            
            # Verificar que a forma é preservada
            assert decompressed_tensor.shape == original_tensor.shape
            assert decompressed_tensor.dtype == original_tensor.dtype


class TestMemoryCompressionManager:
    """Testes para o gerenciador de compressão de memória"""
    
    def test_compression_manager_basic_operation(self):
        """Testa operações básicas do gerenciador de compressão"""
        manager = MemoryCompressionManager()
        
        # Criar tensor de exemplo
        original_tensor = torch.randn(20, 20, dtype=torch.float32)
        
        # Comprimir
        compressed_tensor = manager.compress_tensor(original_tensor)
        
        # Verificar que o tensor foi comprimido
        assert compressed_tensor is not None
        
        # Descomprimir
        decompressed_tensor = manager.decompress_tensor(compressed_tensor)
        
        # Verificar que a forma é preservada
        assert decompressed_tensor.shape == original_tensor.shape
        assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_compression_manager_with_different_tensor_types(self):
        """Testa o gerenciador com diferentes tipos de tensores"""
        manager = MemoryCompressionManager()
        
        # Testar com diferentes tipos e formas
        test_tensors = [
            torch.randn(10),           # 1D
            torch.randn(10, 15),       # 2D
            torch.randn(5, 10, 8),     # 3D
            torch.randn(2, 3, 4, 5),   # 4D
        ]
        
        for original_tensor in test_tensors:
            compressed_tensor = manager.compress_tensor(original_tensor)
            decompressed_tensor = manager.decompress_tensor(compressed_tensor)
            
            # Verificar que a forma e tipo são preservados
            assert decompressed_tensor.shape == original_tensor.shape
            assert decompressed_tensor.dtype == original_tensor.dtype
    
    def test_compression_manager_stats(self):
        """Testa coleta de estatísticas do gerenciador"""
        manager = MemoryCompressionManager()
        
        # Criar tensor e comprimir
        original_tensor = torch.randn(30, 30, dtype=torch.float32)
        compressed_tensor = manager.compress_tensor(original_tensor)
        decompressed_tensor = manager.decompress_tensor(compressed_tensor)
        
        # Obter estatísticas
        stats = manager.get_compression_stats()
        
        # Verificar que as estatísticas têm a estrutura esperada
        assert hasattr(stats, 'compression_ratio')
        assert hasattr(stats, 'compression_time')
        assert hasattr(stats, 'decompression_time')
        assert hasattr(stats, 'memory_saved_bytes')
        assert hasattr(stats, 'total_tensors_compressed')
        
        # Verificar que os valores são razoáveis
        assert stats.compression_ratio >= 0
        assert stats.compression_time >= 0
        assert stats.decompression_time >= 0
        assert stats.memory_saved_bytes >= 0
        assert stats.total_tensors_compressed >= 0
    
    def test_compression_manager_selective_compression(self):
        """Testa compressão seletiva baseada em características do tensor"""
        manager = MemoryCompressionManager()
        
        # Criar tensor que deve ser comprimido (ex: esparsa)
        sparse_tensor = torch.zeros(20, 20, dtype=torch.float32)
        sparse_tensor[::5, ::5] = torch.randn(4, 4)
        
        # Criar tensor que não deve ser comprimido (ex: denso com alta variância)
        dense_tensor = torch.randn(20, 20, dtype=torch.float32)
        
        # Comprimir ambos
        compressed_sparse = manager.compress_tensor(sparse_tensor)
        compressed_dense = manager.compress_tensor(dense_tensor)
        
        # Verificar que ambos foram processados
        assert compressed_sparse is not None
        assert compressed_dense is not None
        
        # Descomprimir ambos
        decompressed_sparse = manager.decompress_tensor(compressed_sparse)
        decompressed_dense = manager.decompress_tensor(compressed_dense)
        
        # Verificar que as formas são preservadas
        assert decompressed_sparse.shape == sparse_tensor.shape
        assert decompressed_dense.shape == dense_tensor.shape
    
    def test_compression_manager_thread_safety(self):
        """Testa segurança em threads do gerenciador"""
        manager = MemoryCompressionManager()
        
        # Função para executar em thread
        def compress_decompress_task(tensor, results, idx):
            compressed = manager.compress_tensor(tensor)
            decompressed = manager.decompress_tensor(compressed)
            results[idx] = torch.allclose(tensor, decompressed, atol=1e-2, rtol=1e-2)
        
        # Criar tensores para diferentes threads
        tensors = [torch.randn(10, 10) for _ in range(5)]
        results = [False] * 5
        threads = []
        
        # Criar e iniciar threads
        for i, tensor in enumerate(tensors):
            thread = threading.Thread(target=compress_decompress_task, args=(tensor, results, i))
            threads.append(thread)
            thread.start()
        
        # Aguardar conclusão
        for thread in threads:
            thread.join()
        
        # Verificar que todas as operações foram bem-sucedidas
        assert all(results)
    
    def test_compression_manager_cache_operations(self):
        """Testa operações de cache do gerenciador"""
        manager = MemoryCompressionManager()
        
        # Criar tensor e comprimir
        original_tensor = torch.randn(25, 25, dtype=torch.float32)
        
        # Armazenar no cache
        key = "test_tensor"
        manager.cache_tensor(key, original_tensor)
        
        # Recuperar do cache
        cached_tensor = manager.get_cached_tensor(key)
        
        # Verificar que o tensor foi recuperado corretamente
        assert cached_tensor is not None
        # Usar allclose com tolerância maior para acomodar diferentes métodos de compressão
        # que podem introduzir diferentes níveis de perda
        assert torch.allclose(original_tensor, cached_tensor, atol=1e-2)
        
        # Remover do cache
        manager.remove_cached_tensor(key)
        
        # Tentar recuperar novamente (deve retornar None)
        missing_tensor = manager.get_cached_tensor(key)
        assert missing_tensor is None


class TestCompressionStats:
    """Testes para a classe de estatísticas de compressão"""
    
    def test_compression_stats_initialization(self):
        """Testa inicialização das estatísticas de compressão"""
        stats = CompressionStats()
        
        # Verificar valores iniciais
        assert stats.compression_ratio == 0.0
        assert stats.compression_time == 0.0
        assert stats.decompression_time == 0.0
        assert stats.memory_saved_bytes == 0
        assert stats.total_tensors_compressed == 0
        assert stats.total_compression_attempts == 0
    
    def test_compression_stats_update(self):
        """Testa atualização das estatísticas de compressão"""
        stats = CompressionStats()
        
        # Atualizar estatísticas
        stats.update_compression(0.5, 0.1, 1000, 1)
        stats.update_compression(0.6, 0.15, 1200, 1)
        
        # Verificar valores atualizados
        assert stats.compression_ratio == 0.55  # Média
        assert stats.compression_time == 0.25  # Soma
        assert stats.decompression_time == 0.0  # Ainda 0, não atualizado
        assert stats.memory_saved_bytes == 2200  # Soma
        assert stats.total_tensors_compressed == 2
        assert stats.total_compression_attempts == 2
    
    def test_compression_stats_reset(self):
        """Testa reset das estatísticas de compressão"""
        stats = CompressionStats()
        
        # Atualizar com alguns valores
        stats.update_compression(0.5, 0.1, 1000, 1)
        
        # Resetar
        stats.reset()
        
        # Verificar valores resetados
        assert stats.compression_ratio == 0.0
        assert stats.compression_time == 0.0
        assert stats.decompression_time == 0.0
        assert stats.memory_saved_bytes == 0
        assert stats.total_tensors_compressed == 0
        assert stats.total_compression_attempts == 0


# Executar os testes se este arquivo for executado diretamente
if __name__ == "__main__":
    pytest.main([__file__])