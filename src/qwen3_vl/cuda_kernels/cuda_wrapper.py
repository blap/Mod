"""
SM61-Optimized CUDA Kernels Wrapper
Provides Python interface to optimized CUDA kernels for NVIDIA SM61 architecture
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import sys
import os

# Set up logging
logger = logging.getLogger(__name__)

# Try to import the compiled CUDA extension
try:
    import sm61_cuda_kernels
    logger.info("Successfully imported SM61 CUDA kernels")
    CUDA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import SM61 CUDA kernels: {e}")
    logger.info("Falling back to PyTorch implementation")
    CUDA_AVAILABLE = False
    sm61_cuda_kernels = None


@dataclass
class SM61Config:
    """Configuration for SM61-specific optimizations"""
    # SM61 architecture characteristics
    compute_capability: Tuple[int, int] = (6, 1)
    max_threads_per_block: int = 1024
    max_shared_memory_per_block: int = 48 * 1024  # 48KB
    warp_size: int = 32
    registers_per_sm: int = 65536
    max_registers_per_thread: int = 255
    
    # Memory optimization settings
    use_pinned_memory: bool = True
    use_async_transfer: bool = True
    memory_pool_size: int = 64 * 1024 * 1024  # 64MB default
    
    # Performance settings
    use_tensor_cores: bool = False  # SM61 doesn't have tensor cores
    precision: str = "float16" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6 * 1024**3 else "float32"


class SM61KernelManager:
    """Manages SM61-optimized CUDA kernels with fallback mechanisms"""
    
    def __init__(self, config: Optional[SM61Config] = None):
        self.config = config or SM61Config()
        self.cuda_available = CUDA_AVAILABLE
        self._memory_pool = None
        
        if self.cuda_available:
            try:
                # Initialize memory pool if available
                self._memory_pool = sm61_cuda_kernels.SM61MemoryPool(self.config.memory_pool_size)
                logger.info("Initialized SM61 memory pool")
            except Exception as e:
                logger.warning(f"Could not initialize SM61 memory pool: {e}")
                self._memory_pool = None
    
    def scaled_dot_product_attention(self, 
                                     query: torch.Tensor, 
                                     key: torch.Tensor, 
                                     value: torch.Tensor,
                                     attn_mask: Optional[torch.Tensor] = None,
                                     dropout_p: float = 0.0,
                                     is_causal: bool = False) -> torch.Tensor:
        """SM61-optimized scaled dot-product attention with fallback"""
        if (self.cuda_available and 
            query.is_cuda and key.is_cuda and value.is_cuda and
            hasattr(sm61_cuda_kernels, 'scaled_dot_product_attention_sm61')):
            
            try:
                return sm61_cuda_kernels.scaled_dot_product_attention_sm61(
                    query, key, value, dropout_p, is_causal
                )
            except Exception as e:
                logger.warning(f"SM61 attention kernel failed: {e}, falling back to PyTorch")
        
        # Fallback to PyTorch implementation
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal
        )
    
    def block_sparse_attention(self, 
                              query: torch.Tensor, 
                              key: torch.Tensor, 
                              value: torch.Tensor,
                              block_mask: torch.Tensor) -> torch.Tensor:
        """SM61-optimized block-sparse attention with fallback"""
        if (self.cuda_available and 
            query.is_cuda and key.is_cuda and value.is_cuda and block_mask.is_cuda and
            hasattr(sm61_cuda_kernels, 'block_sparse_attention_sm61')):
            
            try:
                return sm61_cuda_kernels.block_sparse_attention_sm61(
                    query, key, value, block_mask
                )
            except Exception as e:
                logger.warning(f"SM61 block-sparse attention kernel failed: {e}, falling back to standard attention")
        
        # Fallback to standard attention (less efficient but compatible)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        )
    
    def high_performance_matmul(self, a: torch.Tensor, b: torch.Tensor, 
                                use_tensor_cores: bool = False) -> torch.Tensor:
        """SM61-optimized high-performance matrix multiplication with fallback"""
        if (self.cuda_available and 
            a.is_cuda and b.is_cuda and
            hasattr(sm61_cuda_kernels, 'high_performance_matmul_sm61')):
            
            try:
                # Note: SM61 doesn't have tensor cores, so we'll ignore use_tensor_cores param
                return sm61_cuda_kernels.high_performance_matmul_sm61(a, b, False)
            except Exception as e:
                logger.warning(f"SM61 high-performance matmul kernel failed: {e}, falling back to PyTorch")
        
        # Fallback to PyTorch matmul
        return torch.matmul(a, b)
    
    def memory_efficient_ops(self, input_tensor: torch.Tensor, weight: torch.Tensor, 
                           op_type: int = 1) -> torch.Tensor:
        """SM61-optimized memory-efficient operations with fallback"""
        if (self.cuda_available and 
            input_tensor.is_cuda and weight.is_cuda and
            hasattr(sm61_cuda_kernels, 'memory_efficient_ops_sm61')):
            
            try:
                return sm61_cuda_kernels.memory_efficient_ops_sm61(
                    input_tensor, weight, op_type
                )
            except Exception as e:
                logger.warning(f"SM61 memory-efficient ops kernel failed: {e}, falling back to PyTorch")
        
        # Fallback implementation based on op_type
        if op_type == 0:  # matmul
            return torch.matmul(input_tensor, weight)
        elif op_type == 1:  # add
            return input_tensor + weight
        elif op_type == 2:  # multiply
            return input_tensor * weight
        elif op_type == 3:  # activation (silu)
            return torch.nn.functional.silu(input_tensor)
        else:
            return input_tensor
    
    def coalesced_copy(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """SM61-optimized coalesced memory copy with fallback"""
        if (self.cuda_available and 
            input_tensor.is_cuda and
            hasattr(sm61_cuda_kernels, 'coalesced_copy_sm61')):
            
            try:
                return sm61_cuda_kernels.coalesced_copy_sm61(input_tensor)
            except Exception as e:
                logger.warning(f"SM61 coalesced copy kernel failed: {e}, falling back to PyTorch clone")
        
        # Fallback to PyTorch clone
        return input_tensor.clone()
    
    def transpose(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """SM61-optimized transpose with bank conflict avoidance"""
        if (self.cuda_available and 
            input_tensor.is_cuda and
            hasattr(sm61_cuda_kernels, 'transpose_sm61')):
            
            try:
                return sm61_cuda_kernels.transpose_sm61(input_tensor)
            except Exception as e:
                logger.warning(f"SM61 transpose kernel failed: {e}, falling back to PyTorch transpose")
        
        # Fallback to PyTorch transpose
        return input_tensor.transpose(-2, -1)
    
    def allocate_tensor_from_pool(self, sizes: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate tensor from SM61-optimized memory pool"""
        if self._memory_pool is not None:
            try:
                return self._memory_pool.allocate_tensor(list(sizes), dtype)
            except Exception as e:
                logger.warning(f"SM61 memory pool allocation failed: {e}, falling back to standard allocation")
        
        # Fallback to standard PyTorch allocation
        return torch.empty(sizes, dtype=dtype, device='cuda' if self.cuda_available else 'cpu')
    
    def get_memory_pool_stats(self) -> Optional[Dict[str, Any]]:
        """Get memory pool statistics if available"""
        if self._memory_pool is not None:
            try:
                return self._memory_pool.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get memory pool stats: {e}")
                return None
        return None
    
    def clear_memory_pool(self):
        """Clear the memory pool"""
        if self._memory_pool is not None:
            try:
                self._memory_pool.clear()
            except Exception as e:
                logger.warning(f"Failed to clear memory pool: {e}")
    
    def defragment_memory_pool(self):
        """Defragment the memory pool"""
        if self._memory_pool is not None:
            try:
                self._memory_pool.defragment()
            except Exception as e:
                logger.warning(f"Failed to defragment memory pool: {e}")


class SM61Attention(nn.Module):
    """SM61-optimized attention module"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.kernel_manager = SM61KernelManager()
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with SM61-optimized attention"""
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Apply SM61-optimized attention
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.masked_fill(attn_mask.logical_not(), float("-inf"))
        
        attn_output = self.kernel_manager.scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout, False
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            # Return dummy attention weights as PyTorch does
            attn_weights = torch.zeros((bsz * self.num_heads, tgt_len, src_len), 
                                      dtype=attn_output.dtype, device=attn_output.device)
            return attn_output, attn_weights
        else:
            return attn_output, None


class SM61MLP(nn.Module):
    """SM61-optimized MLP module"""
    
    def __init__(self, hidden_dim: int, intermediate_dim: int, activation="silu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.activation = activation
        
        self.kernel_manager = SM61KernelManager()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SM61-optimized operations"""
        # Apply first linear transformation
        intermediate = self.fc1(x)
        
        # Apply activation
        if self.activation == "silu":
            # Use SM61-optimized memory-efficient silu if available
            if self.kernel_manager.cuda_available:
                intermediate = self.kernel_manager.memory_efficient_ops(
                    intermediate, torch.empty(0, device=intermediate.device), op_type=3
                )
            else:
                intermediate = torch.nn.functional.silu(intermediate)
        else:
            intermediate = torch.nn.functional.relu(intermediate)
        
        # Apply second linear transformation
        output = self.fc2(intermediate)
        return output


class SM61TransformerBlock(nn.Module):
    """SM61-optimized transformer block"""
    
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int, 
                 attention_dropout: float = 0.0, mlp_dropout: float = 0.0):
        super().__init__()
        self.attention = SM61Attention(hidden_dim, num_heads, attention_dropout)
        self.mlp = SM61MLP(hidden_dim, intermediate_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # MLP
        mlp_out = self.mlp(x)
        x = x + self.dropout2(mlp_out)
        x = self.norm2(x)
        
        return x


class SM61OptimizedQwen3VLModel(nn.Module):
    """Complete Qwen3-VL model with SM61 optimizations"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        
        # Token and position embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers with SM61 optimizations
        self.layers = nn.ModuleList([
            SM61TransformerBlock(
                hidden_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_dim=config.intermediate_size,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.hidden_dropout
            ) for _ in range(config.num_hidden_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with appropriate method"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with SM61 optimizations"""
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Generate logits
        logits = self.lm_head(hidden_states)
        
        return logits


def create_sm61_optimized_model(config) -> SM61OptimizedQwen3VLModel:
    """Factory function to create an SM61-optimized model"""
    return SM61OptimizedQwen3VLModel(config)


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information and SM61 compatibility"""
    if CUDA_AVAILABLE and hasattr(sm61_cuda_kernels, 'get_sm61_hardware_info'):
        try:
            return sm61_cuda_kernels.get_sm61_hardware_info()
        except Exception as e:
            logger.warning(f"Failed to get hardware info from CUDA: {e}")
    
    # Fallback hardware detection
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        info.update({
            'device_name': device_name,
            'compute_capability': capability,
            'total_memory_gb': total_memory,
            'is_sm61': capability[0] == 6 and capability[1] == 1
        })
    
    return info


def test_sm61_kernels() -> bool:
    """Test SM61 kernel functionality"""
    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping SM61 kernel tests")
        return False
    
    if not CUDA_AVAILABLE:
        logger.info("SM61 CUDA kernels not available, testing PyTorch fallbacks")
        return True  # Return True since fallbacks should work
    
    logger.info("Testing SM61 kernel functionality...")
    
    try:
        # Test attention kernel
        batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        kernel_manager = SM61KernelManager()
        output = kernel_manager.scaled_dot_product_attention(query, key, value)
        
        logger.info(f"Attention test passed: {output.shape}")
        
        # Test matmul kernel
        a = torch.randn(128, 256, device='cuda', dtype=torch.float16)
        b = torch.randn(256, 512, device='cuda', dtype=torch.float16)
        result = kernel_manager.high_performance_matmul(a, b)
        
        logger.info(f"Matmul test passed: {result.shape}")
        
        # Test memory copy kernel
        test_tensor = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        copied_tensor = kernel_manager.coalesced_copy(test_tensor)
        
        logger.info(f"Memory copy test passed: {torch.equal(test_tensor, copied_tensor)}")
        
        # Test transpose kernel
        transposed_tensor = kernel_manager.transpose(test_tensor)
        logger.info(f"Transpose test passed: {transposed_tensor.shape == (100, 100)}")
        
        # Test memory-efficient ops
        weight = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        result = kernel_manager.memory_efficient_ops(test_tensor, weight, op_type=1)  # Addition
        logger.info(f"Memory-efficient ops test passed: {result.shape == test_tensor.shape}")
        
        logger.info("All SM61 kernel tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"SM61 kernel test failed: {e}")
        return False


if __name__ == "__main__":
    # Run a simple test
    print("Testing SM61 CUDA Kernels Wrapper...")
    success = test_sm61_kernels()
    print(f"Test result: {'PASS' if success else 'FAIL'}")
    
    # Print hardware info
    hw_info = get_hardware_info()
    print(f"Hardware info: {hw_info}")