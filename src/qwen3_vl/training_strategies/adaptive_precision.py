"""
Adaptive Precision Computing Module for Qwen3-VL Architecture
Implements adaptive precision computing with layer-specific precision selection
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class PrecisionConfig:
    """Configuration for precision settings."""
    precision_level: str  # 'fp32', 'fp16', 'int8', 'mixed'
    computation_intensity: float = 1.0
    sensitivity_to_precision: float = 0.5
    memory_footprint: float = 1.0
    accuracy_importance: float = 0.5


class AdaptivePrecisionController(nn.Module):
    """
    Controls adaptive precision allocation based on layer requirements
    and hardware constraints.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        
        # Initialize precision sensitivity for each layer
        self.precision_sensitivity = nn.Parameter(
            torch.zeros(self.num_layers), requires_grad=False
        )
        
        # Initialize layer-specific precision levels
        self.layer_precisions = {}
        for layer_idx in range(self.num_layers):
            # Default to FP16 for most layers, FP32 for critical ones
            if layer_idx % 4 == 0:  # Every 4th layer gets FP32 (critical layers)
                self.layer_precisions[layer_idx] = 'fp32'
            else:
                self.layer_precisions[layer_idx] = 'fp16'
        
        # Performance tracking
        self.performance_history = []
        
        # Hardware-specific optimization flags
        self.target_hardware = getattr(config, 'target_hardware', 'generic')
        self.supports_fp16 = True  # Most modern hardware supports FP16
        self.supports_bf16 = getattr(config, 'use_bfloat16', False)
        
    def profile_precision_sensitivity(self, inputs: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """
        Profile how sensitive each layer is to precision changes.
        """
        sensitivity_results = {}
        
        # For each layer, compute the sensitivity to precision changes
        for layer_idx in range(self.num_layers):
            # Simulate different precision levels and measure impact
            with torch.no_grad():
                # FP32 baseline
                fp32_output = self._simulate_layer_forward(inputs, layer_idx, 'fp32')
                
                # FP16 output
                fp16_output = self._simulate_layer_forward(inputs, layer_idx, 'fp16')
                
                # INT8 output (simulated)
                int8_output = self._simulate_layer_forward(inputs, layer_idx, 'int8')
                
                # Calculate relative errors
                fp16_error = torch.mean(torch.abs(fp32_output - fp16_output)).item()
                int8_error = torch.mean(torch.abs(fp32_output - int8_output)).item()
                
                sensitivity_results[layer_idx] = {
                    'fp32_error': 0.0,  # Baseline
                    'fp16_error': fp16_error,
                    'int8_error': int8_error,
                    'sensitivity_score': (fp16_error + int8_error) / 2.0
                }
        
        return sensitivity_results
    
    def _simulate_layer_forward(self, inputs: torch.Tensor, layer_idx: int, precision: str) -> torch.Tensor:
        """
        Simulate forward pass of a layer with specific precision.
        """
        # This is a simplified simulation - in practice, this would involve
        # actual layer computation with specific precision
        if precision == 'fp32':
            # Use full precision
            return inputs.float()
        elif precision == 'fp16':
            # Simulate half precision effects
            return inputs.half().float()
        elif precision == 'int8':
            # Simulate int8 quantization effects
            # Scale to [-127, 127] range and back
            scale = inputs.abs().max() / 127.0
            if scale == 0:
                scale = 1.0
            quantized = torch.clamp(torch.round(inputs / scale), -127, 127)
            return (quantized * scale).float()
        else:
            return inputs
    
    def select_optimal_precision(self, layer_idx: int, requirements: Dict) -> str:
        """
        Select optimal precision for a layer based on requirements.
        """
        computation_intensity = requirements.get('computation_intensity', 0.5)
        sensitivity_to_precision = requirements.get('sensitivity_to_precision', 0.5)
        memory_footprint = requirements.get('memory_footprint', 0.5)
        accuracy_importance = requirements.get('accuracy_importance', 0.5)
        
        # Calculate precision score for each option
        precision_scores = {}
        
        # FP32: Highest accuracy, highest computation/memory cost
        fp32_score = (1.0 - sensitivity_to_precision) * 0.4 + \
                     (1.0 - computation_intensity) * 0.2 + \
                     (1.0 - memory_footprint) * 0.2 + \
                     accuracy_importance * 0.2
        precision_scores['fp32'] = fp32_score
        
        # FP16: Good balance, widely supported
        fp16_score = (1.0 - sensitivity_to_precision * 0.8) * 0.3 + \
                     (1.0 - computation_intensity * 0.7) * 0.3 + \
                     (1.0 - memory_footprint * 0.6) * 0.3 + \
                     accuracy_importance * 0.1
        precision_scores['fp16'] = fp16_score
        
        # INT8: Lowest resource usage, lowest accuracy
        int8_score = (1.0 - sensitivity_to_precision) * 0.1 + \
                     (1.0 - computation_intensity * 0.9) * 0.4 + \
                     (1.0 - memory_footprint * 0.9) * 0.4 + \
                     accuracy_importance * 0.1
        precision_scores['int8'] = int8_score
        
        # Select precision with highest score that's supported by hardware
        best_precision = 'fp16'  # Default
        best_score = -1
        
        for precision, score in precision_scores.items():
            # Check hardware support
            if precision == 'int8' and not self._supports_int8():
                continue
            if precision == 'bf16' and not self.supports_bf16:
                continue
                
            if score > best_score:
                best_score = score
                best_precision = precision
        
        return best_precision
    
    def _supports_int8(self) -> bool:
        """
        Check if the target hardware supports INT8 operations.
        """
        # For NVIDIA SM61 and similar architectures, INT8 is supported
        return self.target_hardware in ['nvidia_sm61', 'nvidia_turing', 'nvidia_ampere', 'generic']
    
    def adjust_precision_dynamically(self, performance_feedback: Dict) -> Dict[int, str]:
        """
        Adjust precision levels based on performance feedback.
        """
        accuracy_drop = performance_feedback.get('accuracy_drop', 0.0)
        critical_layers = performance_feedback.get('critical_layers', [])
        latency_requirements = performance_feedback.get('latency_requirements', 'balanced')
        memory_constraints = performance_feedback.get('memory_constraints', 'balanced')
        
        updated_precisions = {}
        
        for layer_idx in range(self.num_layers):
            current_precision = self.layer_precisions[layer_idx]
            
            # Determine if we need to adjust precision
            needs_higher_precision = (
                layer_idx in critical_layers or 
                accuracy_drop > 0.02 or  # Significant accuracy drop
                latency_requirements == 'accuracy_priority'
            )
            
            needs_lower_precision = (
                memory_constraints == 'constrained' and 
                layer_idx not in critical_layers and
                accuracy_drop < 0.01  # Minimal accuracy drop
            )
            
            if needs_higher_precision:
                # Upgrade precision if possible
                if current_precision == 'int8':
                    updated_precisions[layer_idx] = 'fp16'
                elif current_precision == 'fp16' and accuracy_drop > 0.03:
                    updated_precisions[layer_idx] = 'fp32'
                else:
                    updated_precisions[layer_idx] = current_precision
            elif needs_lower_precision:
                # Downgrade precision to save memory
                if current_precision == 'fp32':
                    updated_precisions[layer_idx] = 'fp16'
                elif current_precision == 'fp16':
                    updated_precisions[layer_idx] = 'int8'
                else:
                    updated_precisions[layer_idx] = current_precision
            else:
                # Keep current precision
                updated_precisions[layer_idx] = current_precision
        
        # Update internal precision tracking
        self.layer_precisions.update(updated_precisions)
        
        # Store performance feedback for future optimization
        self.performance_history.append(performance_feedback)
        if len(self.performance_history) > 10:  # Keep only recent history
            self.performance_history = self.performance_history[-10:]
        
        return updated_precisions
    
    def get_current_precisions(self) -> Dict[int, str]:
        """Get current precision assignments for all layers."""
        return self.layer_precisions.copy()


class LayerWisePrecisionSelector(nn.Module):
    """
    Selects precision for individual layers based on their specific requirements.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        
        # Initialize layer requirement estimators
        self.layer_requirement_estimators = nn.ModuleList([
            self._create_requirement_estimator() 
            for _ in range(self.num_layers)
        ])
    
    def _create_requirement_estimator(self) -> nn.Sequential:
        """Create a small network to estimate layer requirements."""
        return nn.Sequential(
            nn.Linear(4, 16),  # Input: [computation_intensity, sensitivity, memory_footprint, accuracy_importance]
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)   # Output: weights for precision selection
        )
    
    def estimate_layer_requirements(self, layer_idx: int, inputs: torch.Tensor) -> Dict:
        """Estimate requirements for a specific layer."""
        # Get input statistics
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Estimate computation intensity (based on input complexity)
        input_complexity = torch.std(inputs, dim=-1).mean().item()
        computation_intensity = min(input_complexity / 2.0, 1.0)  # Normalize to [0, 1]
        
        # Estimate sensitivity (based on position and layer importance)
        position_sensitivity = min(layer_idx / self.num_layers, 1.0)
        sensitivity_to_precision = max(0.1, position_sensitivity * 0.7 + 0.3)
        
        # Estimate memory footprint (based on sequence length and batch size)
        memory_footprint = min((seq_len * batch_size) / 1000.0, 1.0)  # Normalize
        
        # Estimate accuracy importance (based on layer position and task)
        accuracy_importance = 0.5  # Default
        if layer_idx < 2:  # Early layers might be more important
            accuracy_importance = 0.8
        elif layer_idx > self.num_layers - 3:  # Late layers might be more important
            accuracy_importance = 0.7
        
        return {
            'computation_intensity': computation_intensity,
            'sensitivity_to_precision': sensitivity_to_precision,
            'memory_footprint': memory_footprint,
            'accuracy_importance': accuracy_importance
        }
    
    def select_precision(self, layer_idx: int, requirements: Dict) -> str:
        """Select precision for a layer based on its requirements."""
        computation_intensity = requirements['computation_intensity']
        sensitivity_to_precision = requirements['sensitivity_to_precision']
        memory_footprint = requirements['memory_footprint']
        accuracy_importance = requirements['accuracy_importance']
        
        # Calculate scores for each precision level
        scores = {}
        
        # FP32 score (high accuracy, high cost)
        fp32_score = (
            accuracy_importance * 0.4 +
            (1.0 - sensitivity_to_precision) * 0.3 +
            (1.0 - memory_footprint) * 0.2 +
            (1.0 - computation_intensity) * 0.1
        )
        scores['fp32'] = fp32_score
        
        # FP16 score (balanced)
        fp16_score = (
            accuracy_importance * 0.3 +
            (1.0 - sensitivity_to_precision * 0.8) * 0.3 +
            (1.0 - memory_footprint * 0.7) * 0.2 +
            (1.0 - computation_intensity * 0.8) * 0.2
        )
        scores['fp16'] = fp16_score
        
        # INT8 score (low cost, lower accuracy)
        int8_score = (
            accuracy_importance * 0.1 +
            (1.0 - sensitivity_to_precision) * 0.1 +
            (1.0 - memory_footprint * 0.9) * 0.5 +
            (1.0 - computation_intensity * 0.9) * 0.3
        )
        scores['int8'] = int8_score
        
        # Mixed precision (for certain operations)
        mixed_score = (
            accuracy_importance * 0.35 +
            (1.0 - sensitivity_to_precision * 0.6) * 0.25 +
            (1.0 - memory_footprint * 0.5) * 0.25 +
            (1.0 - computation_intensity * 0.7) * 0.15
        )
        scores['mixed'] = mixed_score
        
        # Select precision with highest score
        best_precision = max(scores, key=scores.get)
        return best_precision


class PrecisionAdaptiveLayer(nn.Module):
    """
    A wrapper that applies precision adaptation to a neural network layer.
    """
    def __init__(self, layer_module, precision_controller: AdaptivePrecisionController, layer_idx: int):
        super().__init__()
        self.layer = layer_module
        self.precision_controller = precision_controller
        self.layer_idx = layer_idx
        
        # Store the current precision for this layer
        self.current_precision = self.precision_controller.layer_precisions.get(layer_idx, 'fp16')
    
    def forward(self, *args, **kwargs):
        """Forward pass with precision adaptation."""
        # Get current precision for this layer
        precision = self.precision_controller.layer_precisions.get(self.layer_idx, self.current_precision)
        
        # Apply precision transformation if needed
        if precision == 'fp16':
            # Convert inputs to FP16 if supported
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                with torch.cuda.amp.autocast():
                    output = self.layer(*args, **kwargs)
            else:
                # Manual FP16 conversion
                original_dtype = args[0].dtype
                if original_dtype != torch.float16:
                    # Convert inputs to FP16
                    args = [arg.half() if isinstance(arg, torch.Tensor) and arg.dtype != torch.int64 else arg for arg in args]
                    kwargs = {k: v.half() if isinstance(v, torch.Tensor) and v.dtype != torch.int64 else v for k, v in kwargs.items()}
                    
                    output = self.layer(*args, **kwargs)
                    
                    # Convert output back to original dtype
                    if isinstance(output, tuple):
                        output = tuple(
                            o.float() if isinstance(o, torch.Tensor) and o.dtype == torch.float16 else o 
                            for o in output
                        )
                    elif isinstance(output, torch.Tensor):
                        output = output.float()
                else:
                    output = self.layer(*args, **kwargs)
        elif precision == 'int8':
            # Apply INT8 quantization simulation
            output = self._apply_int8_quantization(args, kwargs)
        else:  # FP32 or mixed
            output = self.layer(*args, **kwargs)
        
        return output
    
    def _apply_int8_quantization(self, args, kwargs):
        """Apply INT8 quantization to the layer."""
        # This is a simplified INT8 quantization simulation
        # In practice, this would use proper quantization libraries
        
        # Store original parameters
        original_params = {}
        for name, param in self.layer.named_parameters():
            original_params[name] = param.clone()
            
            # Quantize parameters to INT8 range
            param_max = param.abs().max()
            if param_max == 0:
                param_max = 1.0
            scale = param_max / 127.0
            quantized_param = torch.clamp(torch.round(param / scale), -127, 127)
            self.layer._parameters[name] = nn.Parameter(quantized_param * scale)
        
        # Forward pass with quantized parameters
        output = self.layer(*args, **kwargs)
        
        # Restore original parameters
        for name, original_param in original_params.items():
            self.layer._parameters[name] = nn.Parameter(original_param)
        
        return output


class AdaptivePrecisionAttention(nn.Module):
    """
    Attention mechanism with adaptive precision based on layer requirements.
    """
    def __init__(self, config, layer_idx: int, precision_controller: AdaptivePrecisionController):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.precision_controller = precision_controller
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Core attention components
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embedding
        self.rotary_emb = Qwen3VLRotaryEmbedding(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Get precision for this layer
        precision = self.precision_controller.layer_precisions.get(self.layer_idx, 'fp16')

        # Apply precision-specific computation
        if precision == 'int8':
            # Use INT8-optimized attention computation
            return self._forward_int8(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, cache_position
            )
        elif precision == 'fp16':
            # Use FP16-optimized attention computation
            return self._forward_fp16(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, cache_position
            )
        else:  # FP32
            # Use standard attention computation
            return self._forward_fp32(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, cache_position
            )
    
    def _forward_fp32(self, *args, **kwargs):
        """Standard FP32 attention forward pass."""
        # This is the standard implementation
        hidden_states = args[0]
        attention_mask = args[3] if len(args) > 3 else kwargs.get('attention_mask')
        position_ids = args[2] if len(args) > 2 else kwargs.get('position_ids')
        past_key_value = args[4] if len(args) > 4 else kwargs.get('past_key_value')
        output_attentions = args[5] if len(args) > 5 else kwargs.get('output_attentions', False)
        use_cache = args[6] if len(args) > 6 else kwargs.get('use_cache', False)
        cache_position = args[7] if len(args) > 7 else kwargs.get('cache_position')

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Handle position IDs generation if not provided
        if position_ids is None:
            if past_key_value is not None:
                # If we have past key values, position_ids should account for that
                seq_len = q_len + past_key_value[0].shape[-2]
                position_ids = torch.arange(seq_len - q_len, seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            else:
                # Otherwise, generate position IDs from 0 to q_len
                position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value
    
    def _forward_fp16(self, *args, **kwargs):
        """FP16-optimized attention forward pass."""
        # Use autocast for FP16 operations
        with torch.cuda.amp.autocast():
            return self._forward_fp32(*args, **kwargs)
    
    def _forward_int8(self, *args, **kwargs):
        """INT8-simulated attention forward pass."""
        # This is a simplified INT8 simulation
        # In practice, proper INT8 kernels would be used

        hidden_states = args[0]
        attention_mask = args[3] if len(args) > 3 else kwargs.get('attention_mask')
        position_ids = args[2] if len(args) > 2 else kwargs.get('position_ids')
        past_key_value = args[4] if len(args) > 4 else kwargs.get('past_key_value')
        output_attentions = args[5] if len(args) > 5 else kwargs.get('output_attentions', False)
        use_cache = args[6] if len(args) > 6 else kwargs.get('use_cache', False)
        cache_position = args[7] if len(args) > 7 else kwargs.get('cache_position')

        bsz, q_len, _ = hidden_states.size()

        # Handle position IDs generation if not provided
        if position_ids is None:
            if past_key_value is not None:
                # If we have past key values, position_ids should account for that
                seq_len = q_len + past_key_value[0].shape[-2]
                position_ids = torch.arange(seq_len - q_len, seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            else:
                # Otherwise, generate position IDs from 0 to q_len
                position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        # Quantize inputs for INT8 processing
        hidden_states_quant = self._quantize_tensor(hidden_states)

        # Apply quantized projections
        query_states = self._quantize_tensor(self.q_proj(self._dequantize_tensor(hidden_states_quant)))
        key_states = self._quantize_tensor(self.k_proj(self._dequantize_tensor(hidden_states_quant)))
        value_states = self._quantize_tensor(self.v_proj(self._dequantize_tensor(hidden_states_quant)))

        # Reshape for multi-head attention
        query_states = self._dequantize_tensor(query_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self._dequantize_tensor(key_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self._dequantize_tensor(value_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32 for stability, then convert back
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to INT8 range."""
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            scale = 1.0
        return torch.clamp(torch.round(tensor / scale), -127, 127), scale
    
    def _dequantize_tensor(self, quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor from INT8 range."""
        if isinstance(quantized_tensor, tuple):
            quantized_tensor, scale = quantized_tensor
        return quantized_tensor * scale


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3VLRotaryEmbedding(nn.Module):
    """
    Rotary Embedding implementation for Qwen3-VL.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed