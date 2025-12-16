"""
INT8 Quantization Optimization for Qwen3-VL Model
Implementation of INT8 quantization techniques for CPU optimization on Intel i5-10210U

This module implements INT8 quantization for the Qwen3-VL model to reduce memory usage
and improve inference speed on CPU, specifically targeting the Intel i5-10210U architecture.
"""
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.qconfig import default_qconfig, default_qat_qconfig
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import time
from dataclasses import dataclass
import logging


@dataclass
class INT8QuantizationConfig:
    """Configuration for INT8 quantization optimization."""
    # Quantization parameters
    quantization_mode: str = "static"  # 'static', 'dynamic', or 'qat' (quantization aware training)
    activation_bits: int = 8
    weight_bits: int = 8
    bias_bits: int = 32  # Keep bias at higher precision
    
    # Calibration parameters
    calibration_samples: int = 100
    calibration_method: str = "histogram"  # 'minmax', 'histogram', or 'percentile'
    
    # Performance optimization
    enable_fusion: bool = True  # Enable layer fusion for better performance
    target_platform: str = "intel_cpu"  # 'intel_cpu', 'arm', 'generic'
    
    # Model-specific parameters
    quantize_embeddings: bool = True  # Whether to quantize embedding layers
    quantize_attention: bool = True   # Whether to quantize attention components
    quantize_mlp: bool = True         # Whether to quantize MLP components
    quantize_ln: bool = False         # Whether to quantize layer norms (usually kept in FP32)


class INT8QuantizedAttention(nn.Module):
    """
    Quantized attention mechanism with INT8 operations.
    Implements INT8 quantization for query, key, and value projections.
    """
    def __init__(self, config, layer_idx: Optional[int] = None, quantization_config: INT8QuantizationConfig = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.quantization_config = quantization_config or INT8QuantizationConfig()

        # Use appropriate attributes from the config depending on its type
        if hasattr(config, 'hidden_size'):
            self.hidden_size = config.hidden_size
        else:
            self.hidden_size = config.text_config.hidden_size if hasattr(config, 'text_config') else 512

        if hasattr(config, 'num_attention_heads'):
            self.num_heads = config.num_attention_heads
        else:
            self.num_heads = config.text_config.num_attention_heads if hasattr(config, 'text_config') else 8

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections with INT8 quantization support
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Quantization stubs for INT8 operations
        if self.quantization_config.quantize_attention:
            self.quant_q = QuantStub()
            self.dequant_q = DeQuantStub()
            self.quant_k = QuantStub()
            self.dequant_k = DeQuantStub()
            self.quant_v = QuantStub()
            self.dequant_v = DeQuantStub()
            self.quant_o = QuantStub()
            self.dequant_o = DeQuantStub()
        
        # Rotary embeddings (keep in FP32 for accuracy)
        from advanced_cpu_optimizations_intel_i5_10210u import IntelRotaryEmbedding
        self.rotary_emb = IntelRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Cache for optimized attention computation
        self.scale = (self.head_dim ** -0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Apply projections
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Update cache with new keys and values
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Apply INT8 quantization to attention computation
        if self.quantization_config.quantize_attention:
            # Quantize query, key, and value tensors
            query_states = self.quant_q(query_states)
            key_states = self.quant_k(key_states)
            value_states = self.quant_v(value_states)

            # Dequantize for attention computation
            query_states = self.dequant_q(query_states)
            key_states = self.dequant_k(key_states)
            value_states = self.dequant_v(value_states)

        # Compute attention scores using optimized operations
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax using optimized operations
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # Apply output projection with quantization
        if self.quantization_config.quantize_attention:
            attn_output = self.quant_o(attn_output)
            attn_output = self.dequant_o(attn_output)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        """Apply rotary position embeddings to query and key tensors."""
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat the key and value tensors n_rep times along the head dimension.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class INT8QuantizedMLP(nn.Module):
    """
    Quantized MLP layer with INT8 operations.
    Implements INT8 quantization for gate, up, and down projections.
    """
    def __init__(self, config, quantization_config: INT8QuantizationConfig = None):
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config or INT8QuantizationConfig()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Standard MLP components
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU()

        # Quantization stubs for INT8 operations
        if self.quantization_config.quantize_mlp:
            self.quant_gate = QuantStub()
            self.dequant_gate = DeQuantStub()
            self.quant_up = QuantStub()
            self.dequant_up = DeQuantStub()
            self.quant_down = QuantStub()
            self.dequant_down = DeQuantStub()
            self.quant_intermediate = QuantStub()
            self.dequant_intermediate = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize inputs if enabled
        if self.quantization_config.quantize_mlp:
            x = self.quant_gate(x)
            x = self.dequant_gate(x)

        # Use optimized computation with proper memory layout
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        # Apply activation function
        activated_gate = self.act_fn(gate_output)

        # Element-wise multiplication with optimized memory access
        intermediate_output = activated_gate * up_output

        # Quantize intermediate output if enabled
        if self.quantization_config.quantize_mlp:
            intermediate_output = self.quant_intermediate(intermediate_output)
            intermediate_output = self.dequant_intermediate(intermediate_output)

        # Down projection with quantization
        if self.quantization_config.quantize_mlp:
            intermediate_output = self.quant_down(intermediate_output)
            intermediate_output = self.dequant_down(intermediate_output)

        output = self.down_proj(intermediate_output)

        return output


class INT8QuantizedEmbedding(nn.Module):
    """
    Quantized embedding layer with INT8 operations.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, quantization_config: INT8QuantizationConfig = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quantization_config = quantization_config or INT8QuantizationConfig()
        
        # Standard embedding layer
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)
        
        # Quantization stubs
        if self.quantization_config.quantize_embeddings:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Standard embedding lookup
        output = F.embedding(input_ids, self.weight)
        
        # Apply quantization if enabled
        if self.quantization_config.quantize_embeddings:
            output = self.quant(output)
            output = self.dequant(output)
        
        return output


class INT8QuantizedDecoderLayer(nn.Module):
    """
    Transformer decoder layer with INT8 quantization.
    Combines quantized attention and MLP with quantized normalization.
    """
    def __init__(self, config, layer_idx: int, quantization_config: INT8QuantizationConfig = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.quantization_config = quantization_config or INT8QuantizationConfig()

        # Initialize INT8-quantized submodules
        self.self_attn = INT8QuantizedAttention(config, layer_idx, quantization_config)

        # Use quantized MLP
        self.mlp = INT8QuantizedMLP(config, quantization_config)

        # Normalization layers (usually kept in FP32 for stability)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Quantization stubs for residual connections if needed
        if self.quantization_config.quantize_ln:
            self.quant_input = QuantStub()
            self.dequant_input = DeQuantStub()
            self.quant_post_attn = QuantStub()
            self.dequant_post_attn = DeQuantStub()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Apply input layer norm
        residual = hidden_states
        
        # Quantize input if enabled
        if self.quantization_config.quantize_ln:
            hidden_states = self.quant_input(hidden_states)
            hidden_states = self.dequant_input(hidden_states)
        
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention with INT8 optimizations
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # Add residual connection
        hidden_states = residual + attn_output

        # Apply post-attention layer norm
        residual = hidden_states
        
        # Quantize post-attention input if enabled
        if self.quantization_config.quantize_ln:
            hidden_states = self.quant_post_attn(hidden_states)
            hidden_states = self.dequant_post_attn(hidden_states)
        
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP with INT8 optimizations
        feed_forward_hidden_states = self.mlp(hidden_states)

        # Add residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class INT8Quantizer:
    """
    Main quantizer class that applies INT8 quantization to the Qwen3-VL model.
    """
    def __init__(self, quantization_config: INT8QuantizationConfig = None):
        self.config = quantization_config or INT8QuantizationConfig()
        self.logger = logging.getLogger(__name__)

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply INT8 quantization to the model.
        
        Args:
            model: The original model to quantize
            
        Returns:
            Quantized model
        """
        self.logger.info("Starting INT8 quantization of the model...")
        
        # First, prepare the model for quantization
        model = self._prepare_model_for_quantization(model)
        
        # Then, convert the prepared model to a quantized version
        quantized_model = self._convert_to_quantized_model(model)
        
        self.logger.info("INT8 quantization completed successfully!")
        return quantized_model

    def _prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """
        Prepare the model for quantization by replacing layers with quantizable versions.
        """
        # Replace transformer layers with INT8-quantized versions if the model has the expected structure
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            for layer_idx, layer in enumerate(model.language_model.layers):
                # Check if layer has the expected attention and MLP components
                if hasattr(layer, 'self_attn') and hasattr(layer, 'mlp'):
                    original_attn = layer.self_attn
                    original_mlp = layer.mlp

                    # Check if original attention has the expected projections
                    if (hasattr(original_attn, 'q_proj') and
                        hasattr(original_attn, 'k_proj') and
                        hasattr(original_attn, 'v_proj') and
                        hasattr(original_attn, 'o_proj')):

                        # Create INT8-quantized attention
                        quantized_attn = INT8QuantizedAttention(
                            self._get_attention_config(original_attn),
                            layer_idx=layer_idx,
                            quantization_config=self.config
                        )

                        # Copy parameters from original to quantized attention if possible
                        try:
                            quantized_attn.q_proj.weight.data = original_attn.q_proj.weight.data.clone()
                            quantized_attn.k_proj.weight.data = original_attn.k_proj.weight.data.clone()
                            quantized_attn.v_proj.weight.data = original_attn.v_proj.weight.data.clone()
                            quantized_attn.o_proj.weight.data = original_attn.o_proj.weight.data.clone()
                        except Exception as e:
                            self.logger.warning(f"Could not copy attention parameters for layer {layer_idx}: {e}")
                            self.logger.warning("Using random initialization for quantized attention")

                        # Check if original MLP has the expected projections
                        if (hasattr(original_mlp, 'gate_proj') and
                            hasattr(original_mlp, 'up_proj') and
                            hasattr(original_mlp, 'down_proj')):

                            # Create INT8-quantized MLP
                            quantized_mlp = INT8QuantizedMLP(
                                self._get_mlp_config(original_mlp),
                                quantization_config=self.config
                            )

                            # Copy parameters from original to quantized MLP if possible
                            try:
                                quantized_mlp.gate_proj.weight.data = original_mlp.gate_proj.weight.data.clone()
                                quantized_mlp.up_proj.weight.data = original_mlp.up_proj.weight.data.clone()
                                quantized_mlp.down_proj.weight.data = original_mlp.down_proj.weight.data.clone()
                            except Exception as e:
                                self.logger.warning(f"Could not copy MLP parameters for layer {layer_idx}: {e}")
                                self.logger.warning("Using random initialization for quantized MLP")

                            # Replace the layers
                            layer.self_attn = quantized_attn
                            layer.mlp = quantized_mlp
                        else:
                            self.logger.info(f"Skipping MLP quantization for layer {layer_idx}, missing expected projections")
                    else:
                        self.logger.info(f"Skipping attention quantization for layer {layer_idx}, missing expected projections")
                else:
                    self.logger.info(f"Skipping layer {layer_idx}, missing expected components")

        # Quantize embeddings if specified
        if self.config.quantize_embeddings:
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                original_embeddings = model.model.embed_tokens
                quantized_embeddings = INT8QuantizedEmbedding(
                    original_embeddings.num_embeddings,
                    original_embeddings.embedding_dim,
                    self.config
                )
                # Copy weights
                quantized_embeddings.weight.data = original_embeddings.weight.data.clone()
                model.model.embed_tokens = quantized_embeddings
            elif hasattr(model, 'embed_tokens'):
                original_embeddings = model.embed_tokens
                quantized_embeddings = INT8QuantizedEmbedding(
                    original_embeddings.num_embeddings,
                    original_embeddings.embedding_dim,
                    self.config
                )
                # Copy weights
                quantized_embeddings.weight.data = original_embeddings.weight.data.clone()
                model.embed_tokens = quantized_embeddings

        # Set the model to evaluation mode for quantization
        model.eval()
        
        # Prepare the model for static quantization
        if self.config.quantization_mode == "static":
            model.qconfig = default_qconfig
            torch.quantization.prepare(model, inplace=True)
        
        return model

    def _convert_to_quantized_model(self, model: nn.Module) -> nn.Module:
        """
        Convert the prepared model to a fully quantized model.
        """
        if self.config.quantization_mode == "static":
            # Convert the prepared model to a quantized model
            model = torch.quantization.convert(model, inplace=True)
        elif self.config.quantization_mode == "dynamic":
            # For dynamic quantization, we quantize weights only at runtime
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
        
        return model

    def _get_attention_config(self, attention_layer):
        """
        Extract configuration from the original attention layer.
        """
        # Create a mock config with the necessary attributes
        class MockConfig:
            pass
        
        mock_config = MockConfig()
        mock_config.hidden_size = attention_layer.q_proj.out_features
        mock_config.num_attention_heads = mock_config.hidden_size // attention_layer.q_proj.out_features
        mock_config.num_key_value_heads = getattr(attention_layer, "num_key_value_heads", mock_config.num_attention_heads)
        mock_config.max_position_embeddings = getattr(attention_layer, "max_position_embeddings", 2048)
        mock_config.rope_theta = getattr(attention_layer, "rope_theta", 10000.0)
        mock_config.layer_norm_eps = getattr(attention_layer, "layer_norm_eps", 1e-5)
        
        return mock_config

    def _get_mlp_config(self, mlp_layer):
        """
        Extract configuration from the original MLP layer.
        """
        # Create a mock config with the necessary attributes
        class MockConfig:
            pass
        
        mock_config = MockConfig()
        mock_config.hidden_size = mlp_layer.gate_proj.out_features
        mock_config.intermediate_size = mlp_layer.up_proj.out_features
        mock_config.layer_norm_eps = getattr(mlp_layer, "layer_norm_eps", 1e-5)
        
        return mock_config

    def calibrate_model(self, model: nn.Module, calibration_data_loader) -> nn.Module:
        """
        Calibrate the quantized model using calibration data.
        
        Args:
            model: The quantized model to calibrate
            calibration_data_loader: DataLoader with calibration data
            
        Returns:
            Calibrated model
        """
        self.logger.info("Starting model calibration...")
        
        # Set the model to evaluation mode
        model.eval()
        
        # Perform calibration
        with torch.no_grad():
            for i, batch in enumerate(calibration_data_loader):
                if i >= self.config.calibration_samples:
                    break
                
                # Run the model with the calibration batch
                try:
                    if isinstance(batch, dict):
                        _ = model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = model(*batch)
                    else:
                        # Assume it's input_ids for language models
                        _ = model(input_ids=batch)
                except Exception as e:
                    self.logger.warning(f"Calibration failed for batch {i}: {e}")
                    continue
        
        self.logger.info("Model calibration completed!")
        return model

    def benchmark_quantization_impact(self, original_model: nn.Module, quantized_model: nn.Module, 
                                    test_data_loader) -> Dict[str, Any]:
        """
        Benchmark the impact of quantization on model performance.
        
        Args:
            original_model: The original model
            quantized_model: The quantized model
            test_data_loader: DataLoader with test data
            
        Returns:
            Dictionary with performance metrics
        """
        # Set both models to evaluation mode
        original_model.eval()
        quantized_model.eval()
        
        # Track performance metrics
        original_times = []
        quantized_times = []
        original_memory = []
        quantized_memory = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                if i >= 10:  # Limit to 10 batches for quick benchmarking
                    break
                
                # Benchmark original model
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                else:
                    start_time_cpu = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.time()
                
                try:
                    if isinstance(batch, dict):
                        _ = original_model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = original_model(*batch)
                    else:
                        _ = original_model(input_ids=batch)
                except Exception:
                    continue
                
                if start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    original_time = start_time.elapsed_time(end_time)
                else:
                    end_time_cpu = time.time()
                    original_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to milliseconds
                
                original_times.append(original_time)
                
                # Benchmark quantized model
                if start_time:
                    start_time.record()
                
                try:
                    if isinstance(batch, dict):
                        _ = quantized_model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = quantized_model(*batch)
                    else:
                        _ = quantized_model(input_ids=batch)
                except Exception:
                    continue
                
                if start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    quantized_time = start_time.elapsed_time(end_time)
                else:
                    start_time_cpu = time.time()
                    _ = quantized_model(input_ids=batch)
                    end_time_cpu = time.time()
                    quantized_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to milliseconds
                
                quantized_times.append(quantized_time)
        
        # Calculate metrics
        avg_original_time = np.mean(original_times) if original_times else 0
        avg_quantized_time = np.mean(quantized_times) if quantized_times else 0
        speedup = avg_original_time / avg_quantized_time if avg_quantized_time > 0 else 0
        
        # Calculate model sizes
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024**2)  # MB
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024**2)  # MB
        size_reduction = (original_size - quantized_size) / original_size * 100 if original_size > 0 else 0
        
        return {
            'original_avg_time_ms': avg_original_time,
            'quantized_avg_time_ms': avg_quantized_time,
            'speedup': speedup,
            'original_model_size_mb': original_size,
            'quantized_model_size_mb': quantized_size,
            'size_reduction_percent': size_reduction,
            'num_test_batches': len(original_times)
        }


def apply_int8_quantization_to_model(
    model: nn.Module,
    config: Optional[INT8QuantizationConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply INT8 quantization to the Qwen3-VL model.

    Args:
        model: The Qwen3-VL model to quantize
        config: Configuration for INT8 quantization (optional)

    Returns:
        Tuple of (quantized_model, quantization_info)
    """
    logger = logging.getLogger(__name__)
    logger.info("Applying INT8 quantization to the Qwen3-VL model...")

    # Use default config if none provided
    if config is None:
        config = INT8QuantizationConfig()

    # Initialize the quantizer
    quantizer = INT8Quantizer(config)

    # Apply quantization
    quantized_model = quantizer.quantize_model(model)

    # Create quantization info
    quantization_info = {
        'config': config,
        'quantization_mode': config.quantization_mode,
        'weight_bits': config.weight_bits,
        'activation_bits': config.activation_bits,
        'quantize_embeddings': config.quantize_embeddings,
        'quantize_attention': config.quantize_attention,
        'quantize_mlp': config.quantize_mlp,
        'quantize_ln': config.quantize_ln
    }

    logger.info("INT8 quantization applied successfully!")
    return quantized_model, quantization_info


