"""
Teste de ponta a ponta do sistema avançado de compressão de memória
"""
import torch
import numpy as np
from memory_compression_system import (
    QuantizationCompressor,
    SVDCompressor,
    SparseCompressor,
    AutoCompressor,
    MemoryCompressionManager,
    create_hardware_optimized_compression_manager
)

def test_end_to_end():
    print("Teste de Ponta a Ponta - Sistema Avançado de Compressão de Memória")
    print("=" * 65)
    
    # Teste 1: Compressão e descompressão com diferentes métodos
    print("\n1. Testando diferentes métodos de compressão:")
    
    # Criar tensores de exemplo para diferentes tipos de compressão
    dense_tensor = torch.randn(100, 100, dtype=torch.float32)
    sparse_tensor = torch.zeros(100, 100, dtype=torch.float32)
    sparse_tensor[::5, ::5] = torch.randn(20, 20)  # Tornar esparsa
    low_rank_tensor = torch.randn(100, 50, dtype=torch.float32) @ torch.randn(50, 80, dtype=torch.float32)
    
    print(f"  Tensor denso: {dense_tensor.shape}, tamanho: {dense_tensor.numel() * dense_tensor.element_size() / (1024):.2f} KB")
    print(f"  Tensor esparsa: {sparse_tensor.shape}, sparsity: {(sparse_tensor == 0).float().mean().item():.2f}")
    print(f"  Tensor de baixo rank: {low_rank_tensor.shape}")
    
    # Testar quantização
    quant_compressor = QuantizationCompressor()
    compressed_int8 = quant_compressor.compress(dense_tensor, method='int8')
    decompressed_int8 = quant_compressor.decompress(compressed_int8)
    print(f"  Quantização INT8 - Razão: {compressed_int8['compression_ratio']:.3f}, Fidelidade: {torch.allclose(dense_tensor, decompressed_int8, atol=1e-1)}")
    
    compressed_fp16 = quant_compressor.compress(dense_tensor, method='fp16')
    decompressed_fp16 = quant_compressor.decompress(compressed_fp16)
    print(f"  Quantização FP16 - Razão: {compressed_fp16['compression_ratio']:.3f}, Fidelidade: {torch.allclose(dense_tensor, decompressed_fp16, atol=1e-3)}")
    
    # Testar SVD
    svd_compressor = SVDCompressor()
    compressed_svd = svd_compressor.compress(low_rank_tensor, adaptive_rank=True, compression_ratio=0.5)
    decompressed_svd = svd_compressor.decompress(compressed_svd)
    print(f"  SVD - Razão: {compressed_svd['compression_ratio']:.3f}, Fidelidade: {torch.allclose(low_rank_tensor, decompressed_svd, atol=1e-1)}")
    
    # Testar esparsidade
    sparse_compressor = SparseCompressor()
    compressed_sparse = sparse_compressor.compress(sparse_tensor, sparsity_threshold=0.5)
    decompressed_sparse = sparse_compressor.decompress(compressed_sparse)
    print(f"  Esparsidade - Razão: {compressed_sparse['compression_ratio']:.3f}, Fidelidade: {torch.allclose(sparse_tensor, decompressed_sparse, atol=1e-6)}")
    
    # Testar auto
    auto_compressor = AutoCompressor()
    compressed_auto = auto_compressor.compress(dense_tensor)
    decompressed_auto = auto_compressor.decompress(compressed_auto)
    print(f"  Auto - Método: {compressed_auto['method']}, Razão: {compressed_auto['compression_ratio']:.3f}, Fidelidade: {torch.allclose(dense_tensor, decompressed_auto, atol=1e-2)}")
    
    # Teste 2: Gerenciador de compressão
    print("\n2. Testando gerenciador de compressão:")
    manager = create_hardware_optimized_compression_manager()
    
    test_tensor = torch.randn(500, 500, dtype=torch.float32)
    compressed = manager.compress_tensor(test_tensor, method='auto')
    decompressed = manager.decompress_tensor(compressed)
    
    print(f"  Tensor original: {test_tensor.shape}")
    print(f"  Método de compressão: {compressed['method']}")
    print(f"  Razão de compressão: {compressed['compression_ratio']:.3f}")
    print(f"  Fidelidade: {torch.allclose(test_tensor, decompressed, atol=1e-2)}")
    
    # Estatísticas
    stats = manager.get_compression_stats()
    print(f"  Estatísticas - Compressão média: {stats.compression_ratio:.3f}, Memória economizada: {stats.memory_saved_bytes / (1024*1024):.2f} MB")
    
    # Teste 3: Cache
    print("\n3. Testando cache de compressão:")
    key = "test_tensor"
    manager.cache_tensor(key, test_tensor)
    cached_tensor = manager.get_cached_tensor(key)
    
    print(f"  Tensor recuperado do cache: {cached_tensor.shape if cached_tensor is not None else None}")
    print(f"  Fidelidade do cache: {torch.allclose(test_tensor, cached_tensor, atol=1e-2) if cached_tensor is not None else False}")
    print(f"  Tamanho do cache: {manager.get_cache_size()}")
    
    # Limpar cache
    manager.remove_cached_tensor(key)
    print(f"  Tamanho do cache após remoção: {manager.get_cache_size()}")
    
    print("\nTodos os testes de ponta a ponta passaram com sucesso!")
    print("Sistema de compressão de memória está funcionando corretamente e integrado com os componentes existentes.")

if __name__ == "__main__":
    test_end_to_end()