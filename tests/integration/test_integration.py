"""
Teste de integração entre o sistema de pooling de memória e o cache hierárquico avançado
"""

from src.qwen3_vl.components.memory.advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType

def test_integration():
    print("Testando integração entre pooling de memória e cache hierárquico avançado...")
    
    # Criar o sistema de pooling de memória (que inclui o cache hierárquico avançado)
    memory_system = AdvancedMemoryPoolingSystem()
    
    # Alocar alguns tensores de exemplo
    print("Alocando tensores de exemplo...")
    
    # Alocar KV cache (muito acessado)
    kv_block = memory_system.allocate(TensorType.KV_CACHE, 1024*1024*5, "kv_tensor_1")  # 5MB
    if kv_block:
        print(f"Alocado bloco KV cache: {kv_block.size} bytes no endereço {kv_block.start_addr}")
    
    # Alocar features de imagem
    img_block = memory_system.allocate(TensorType.IMAGE_FEATURES, 1024*1024*3, "img_tensor_1")  # 3MB
    if img_block:
        print(f"Alocado bloco features de imagem: {img_block.size} bytes no endereço {img_block.start_addr}")
    
    # Alocar embeddings de texto
    text_block = memory_system.allocate(TensorType.TEXT_EMBEDDINGS, 1024*1024*2, "text_tensor_1")  # 2MB
    if text_block:
        print(f"Alocado bloco embeddings de texto: {text_block.size} bytes no endereço {text_block.start_addr}")
    
    # Mostrar estatísticas do sistema, incluindo do cache hierárquico avançado
    print("\nEstatísticas do sistema (incluindo cache hierárquico avançado):")
    stats = memory_system.get_system_stats()
    for key, value in stats.items():
        if key != 'advanced_cache_stats':
            print(f"  {key}: {value}")
    
    print("\nEstatísticas do cache hierárquico avançado:")
    cache_stats = stats['advanced_cache_stats']
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # Acessar novamente os tensores para testar o cache
    print("\nAcessando novamente os tensores para testar o cache...")
    kv_block_again = memory_system.allocate(TensorType.KV_CACHE, 1024*1024*5, "kv_tensor_1")
    if kv_block_again:
        print(f"Acesso ao KV cache: {kv_block_again.size} bytes")
    
    # Desalocar
    print("\nDesalocando tensores...")
    memory_system.deallocate(TensorType.KV_CACHE, "kv_tensor_1")
    memory_system.deallocate(TensorType.IMAGE_FEATURES, "img_tensor_1")
    memory_system.deallocate(TensorType.TEXT_EMBEDDINGS, "text_tensor_1")
    
    # Compactar memória
    print("\nCompactando memória...")
    memory_system.compact_memory()
    
    # Mostrar estatísticas finais
    print("\nEstatísticas finais do sistema:")
    final_stats = memory_system.get_system_stats()
    cache_stats_final = final_stats['advanced_cache_stats']
    print(f"  Total alocado: {final_stats['total_allocated']}")
    print(f"  Total liberado: {final_stats['total_freed']}")
    print(f"  Hit rate geral do cache: {cache_stats_final.get('overall_hit_rate', 0):.2%}")
    
    print("\nIntegração entre pooling de memória e cache hierárquico avançado testada com sucesso!")

if __name__ == "__main__":
    test_integration()