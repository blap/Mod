"""
Safety Checks for Qwen3-VL Model Capacity Preservation
Ensures all optimizations maintain the full 32 transformer layers and 32 attention heads.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
import warnings
from collections import OrderedDict
from enum import Enum
import time


class SafetyCheckType(Enum):
    """Types of safety checks"""
    LAYER_COUNT = "layer_count"
    ATTENTION_HEADS = "attention_heads"
    PARAMETER_COUNT = "parameter_count"
    MODEL_ARCHITECTURE = "model_architecture"
    FORWARD_PASS_INTEGRITY = "forward_pass_integrity"
    GRADIENT_FLOW = "gradient_flow"


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    check_type: SafetyCheckType
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ModelCapacityValidator:
    """
    Validates that model capacity is preserved through optimization techniques.
    Ensures 32 transformer layers and 32 attention heads are maintained.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.safety_check_results: List[SafetyCheckResult] = []
        self.required_layers = 32
        self.required_attention_heads = 32
    
    def validate_layer_count(self, model: nn.Module) -> SafetyCheckResult:
        """Validate that the model has exactly 32 transformer layers"""
        try:
            # Count transformer layers by looking for specific layer types
            layer_count = 0

            # Look for transformer decoder layers
            for name, module in model.named_modules():
                if 'decoder_layer' in name.lower() or 'transformer_layer' in name.lower() or hasattr(module, 'self_attn'):
                    if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)) or hasattr(module, 'self_attn'):
                        layer_count += 1

            # Alternative: Look for specific Qwen3-VL layer names
            if layer_count == 0:
                for name, _ in model.named_modules():
                    if 'layer_' in name or 'block_' in name:
                        layer_count += 1

            # If we still have 0, count based on parameter groups that typically correspond to layers
            if layer_count == 0:
                for name, param in model.named_parameters():
                    if 'layer' in name or 'block' in name:
                        layer_idx = None
                        try:
                            # Extract layer index from parameter name
                            parts = name.split('.')
                            for part in parts:
                                if 'layer' in part or 'block' in part:
                                    # Try to extract number
                                    for s in part.split('_'):
                                        if s.isdigit():
                                            layer_idx = int(s)
                                            break
                        except:
                            continue
                        if layer_idx is not None:
                            layer_count = max(layer_count, layer_idx + 1)

            # Additional check for Qwen3-VL specific layer patterns
            if layer_count == 0:
                # Look for Qwen3-VL specific patterns
                layer_patterns = ['layers.', 'h.', 'transformer.h.', 'encoder.layers.', 'decoder.layers.']
                for name, module in model.named_modules():
                    for pattern in layer_patterns:
                        if pattern in name:
                            # Extract layer index from pattern like "layers.0", "layers.1", etc.
                            try:
                                # Get the part after the pattern
                                parts = name.split(pattern)
                                if len(parts) > 1 and parts[1]:
                                    # Get the first part and extract number
                                    layer_part = parts[1].split('.')[0]
                                    if layer_part.isdigit():
                                        layer_idx = int(layer_part)
                                        layer_count = max(layer_count, layer_idx + 1)
                            except:
                                continue

            passed = layer_count == self.required_layers
            message = f"Layer count check: Expected {self.required_layers}, found {layer_count}. {'PASSED' if passed else 'FAILED'}"

            result = SafetyCheckResult(
                check_type=SafetyCheckType.LAYER_COUNT,
                passed=passed,
                message=message,
                details={'expected': self.required_layers, 'actual': layer_count}
            )

            self.safety_check_results.append(result)
            return result

        except Exception as e:
            message = f"Layer count validation failed with error: {e}"
            result = SafetyCheckResult(
                check_type=SafetyCheckType.LAYER_COUNT,
                passed=False,
                message=message,
                details={'error': str(e)}
            )
            self.safety_check_results.append(result)
            return result
    
    def validate_attention_heads(self, model: nn.Module) -> SafetyCheckResult:
        """Validate that the model has exactly 32 attention heads per layer"""
        try:
            # Look for attention mechanisms and count heads
            attention_heads_counts = []

            for name, module in model.named_modules():
                if isinstance(module, (nn.MultiheadAttention, nn.Linear)) or 'attention' in name.lower() or 'attn' in name.lower():
                    # Check if this is an attention module
                    if hasattr(module, 'num_heads'):
                        attention_heads_counts.append(module.num_heads)
                    elif hasattr(module, 'in_features') and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                        # Estimate from parameter dimensions (assuming hidden_size = num_heads * head_dim)
                        # For 32 heads with standard head dimensions
                        hidden_size = module.in_features
                        # Common head dimensions are 64, 80, 128 - check for 32 heads
                        possible_heads = [hidden_size // dim for dim in [64, 80, 128] if hidden_size % dim == 0]
                        if self.required_attention_heads in possible_heads:
                            attention_heads_counts.append(self.required_attention_heads)
                    elif hasattr(module, 'in_features') and ('attn' in name and 'c_attn' in name):
                        # For models with combined attention projections like GPT
                        hidden_size = module.in_features
                        # If it's a combined projection (Q, K, V), divide by 3 to get single projection size
                        single_proj_size = hidden_size // 3
                        possible_heads = [single_proj_size // dim for dim in [64, 80, 128] if single_proj_size % dim == 0]
                        if self.required_attention_heads in possible_heads:
                            attention_heads_counts.append(self.required_attention_heads)

            # If we didn't find attention modules directly, try to infer from model config
            if not attention_heads_counts and hasattr(model, 'config'):
                if hasattr(model.config, 'num_attention_heads'):
                    attention_heads_counts = [model.config.num_attention_heads]
                elif hasattr(model.config, 'num_heads'):
                    attention_heads_counts = [model.config.num_heads]
                elif hasattr(model.config, 'n_head'):
                    attention_heads_counts = [model.config.n_head]
                elif hasattr(model.config, 'num_heads'):
                    attention_heads_counts = [model.config.num_heads]

            # For Qwen3-VL, also check for multi-modal attention heads
            if not attention_heads_counts:
                # Check for specific Qwen3-VL attention patterns
                for name, param in model.named_parameters():
                    if 'attn' in name and ('weight' in name or 'bias' in name):
                        # Look for attention-related parameter shapes
                        if param.dim() == 2:  # Weight matrix
                            if param.size(0) % self.required_attention_heads == 0:
                                # If the first dimension is divisible by required heads, it might be attention
                                head_dim = param.size(0) // self.required_attention_heads
                                if 64 <= head_dim <= 128:  # Typical head dimensions
                                    attention_heads_counts.append(self.required_attention_heads)
                                    break

            # Check if all attention mechanisms have the correct number of heads
            if attention_heads_counts:
                all_correct = all(count == self.required_attention_heads for count in attention_heads_counts)
                message = f"Attention heads check: Expected {self.required_attention_heads}, found {attention_heads_counts}. {'PASSED' if all_correct else 'FAILED'}"

                result = SafetyCheckResult(
                    check_type=SafetyCheckType.ATTENTION_HEADS,
                    passed=all_correct,
                    message=message,
                    details={'expected': self.required_attention_heads, 'actual_counts': attention_heads_counts}
                )
            else:
                message = "Attention heads check: Could not determine attention head count"
                result = SafetyCheckResult(
                    check_type=SafetyCheckType.ATTENTION_HEADS,
                    passed=False,
                    message=message,
                    details={'actual_counts': []}
                )

            self.safety_check_results.append(result)
            return result

        except Exception as e:
            message = f"Attention heads validation failed with error: {e}"
            result = SafetyCheckResult(
                check_type=SafetyCheckType.ATTENTION_HEADS,
                passed=False,
                message=message,
                details={'error': str(e)}
            )
            self.safety_check_results.append(result)
            return result
    
    def validate_parameter_count(self, model: nn.Module, baseline_param_count: Optional[int] = None) -> SafetyCheckResult:
        """Validate that the model has not lost significant parameters due to optimizations"""
        try:
            current_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if baseline_param_count is None:
                # If no baseline provided, we'll just report the count
                message = f"Parameter count: {current_param_count:,} parameters"
                result = SafetyCheckResult(
                    check_type=SafetyCheckType.PARAMETER_COUNT,
                    passed=True,  # We pass if we can count parameters
                    message=message,
                    details={'parameter_count': current_param_count}
                )
            else:
                # Check if we've lost more than 5% of parameters (arbitrary threshold)
                threshold = 0.05
                min_acceptable = baseline_param_count * (1 - threshold)
                
                passed = current_param_count >= min_acceptable
                percentage_lost = (baseline_param_count - current_param_count) / baseline_param_count * 100
                
                message = f"Parameter count check: {percentage_lost:.2f}% parameters lost ({current_param_count:,}/{baseline_param_count:,}). {'PASSED' if passed else 'FAILED'}"
                
                result = SafetyCheckResult(
                    check_type=SafetyCheckType.PARAMETER_COUNT,
                    passed=passed,
                    message=message,
                    details={
                        'baseline_count': baseline_param_count,
                        'current_count': current_param_count,
                        'percentage_lost': percentage_lost,
                        'threshold': threshold
                    }
                )
            
            self.safety_check_results.append(result)
            return result
            
        except Exception as e:
            message = f"Parameter count validation failed with error: {e}"
            result = SafetyCheckResult(
                check_type=SafetyCheckType.PARAMETER_COUNT,
                passed=False,
                message=message,
                details={'error': str(e)}
            )
            self.safety_check_results.append(result)
            return result
    
    def validate_model_architecture(self, model: nn.Module, expected_architecture: Optional[Dict[str, Any]] = None) -> SafetyCheckResult:
        """Validate that the model architecture remains intact"""
        try:
            # Get the model's architecture signature
            architecture_signature = self._get_model_signature(model)

            if expected_architecture:
                # Compare with expected architecture
                passed = self._compare_architectures(architecture_signature, expected_architecture)
                message = f"Architecture validation: {'PASSED' if passed else 'FAILED'}"
            else:
                # Validate for Qwen3-VL specific architecture components
                passed = self._validate_qwen3_vl_architecture(model)
                message = f"Qwen3-VL architecture validation: {'PASSED' if passed else 'FAILED'}"

            result = SafetyCheckResult(
                check_type=SafetyCheckType.MODEL_ARCHITECTURE,
                passed=passed,
                message=message,
                details={'architecture_signature': architecture_signature}
            )

            self.safety_check_results.append(result)
            return result

        except Exception as e:
            message = f"Architecture validation failed with error: {e}"
            result = SafetyCheckResult(
                check_type=SafetyCheckType.MODEL_ARCHITECTURE,
                passed=False,
                message=message,
                details={'error': str(e)}
            )
            self.safety_check_results.append(result)
            return result

    def _validate_qwen3_vl_architecture(self, model: nn.Module) -> bool:
        """Validate Qwen3-VL specific architecture components"""
        required_components = [
            'vision_tower', 'visual_projection', 'language_model',
            'multi_modal_projector', 'llm', 'encoder', 'decoder'
        ]

        found_components = []
        for name, module in model.named_modules():
            # Check for Qwen3-VL specific components
            for component in required_components:
                if component in name.lower():
                    found_components.append(component)

        # Check if essential components are present
        essential_components = ['language_model', 'vision_tower']
        essential_found = all(comp in found_components for comp in essential_components)

        # Also check for transformer layers
        layer_count = 0
        for name, module in model.named_modules():
            if 'decoder_layer' in name.lower() or 'transformer_layer' in name.lower() or hasattr(module, 'self_attn'):
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)) or hasattr(module, 'self_attn'):
                    layer_count += 1

        # For Qwen3-VL, we expect at least some transformer layers
        has_layers = layer_count >= 1

        return essential_found and has_layers
    
    def _get_model_signature(self, model: nn.Module) -> Dict[str, Any]:
        """Extract a signature of the model architecture"""
        signature = {
            'module_types': [],
            'layer_names': [],
            'parameter_shapes': [],
            'submodule_count': 0
        }
        
        for name, module in model.named_modules():
            signature['module_types'].append(type(module).__name__)
            signature['layer_names'].append(name)
            signature['submodule_count'] += 1
            
            # Collect parameter shapes
            for param_name, param in module.named_parameters(recurse=False):
                signature['parameter_shapes'].append({
                    'module': name,
                    'param': param_name,
                    'shape': list(param.shape)
                })
        
        return signature
    
    def _compare_architectures(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> bool:
        """Compare two architecture signatures"""
        # Compare module types (order and types should match)
        if sig1['module_types'] != sig2['module_types']:
            return False
        
        # Compare parameter shapes
        if len(sig1['parameter_shapes']) != len(sig2['parameter_shapes']):
            return False
        
        for p1, p2 in zip(sig1['parameter_shapes'], sig2['parameter_shapes']):
            if p1['shape'] != p2['shape']:
                return False
        
        return True
    
    def validate_forward_pass_integrity(self, model: nn.Module, test_input: torch.Tensor) -> SafetyCheckResult:
        """Validate that forward pass works correctly and produces expected output shape"""
        try:
            model.eval()  # Set to evaluation mode
            
            with torch.no_grad():
                try:
                    output = model(test_input)
                    
                    # Check if output is valid (not NaN or inf)
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        passed = False
                        message = "Forward pass integrity: Output contains NaN or Inf values"
                    else:
                        passed = True
                        message = f"Forward pass integrity: Successful, output shape {output.shape}"
                    
                    result = SafetyCheckResult(
                        check_type=SafetyCheckType.FORWARD_PASS_INTEGRITY,
                        passed=passed,
                        message=message,
                        details={'output_shape': list(output.shape), 'has_nan': torch.isnan(output).any().item(), 'has_inf': torch.isinf(output).any().item()}
                    )
                    
                except Exception as e:
                    passed = False
                    message = f"Forward pass failed with error: {e}"
                    result = SafetyCheckResult(
                        check_type=SafetyCheckType.FORWARD_PASS_INTEGRITY,
                        passed=passed,
                        message=message,
                        details={'error': str(e)}
                    )
            
            self.safety_check_results.append(result)
            return result
            
        except Exception as e:
            message = f"Forward pass integrity validation failed with error: {e}"
            result = SafetyCheckResult(
                check_type=SafetyCheckType.FORWARD_PASS_INTEGRITY,
                passed=False,
                message=message,
                details={'error': str(e)}
            )
            self.safety_check_results.append(result)
            return result
    
    def validate_gradient_flow(self, model: nn.Module, test_input: torch.Tensor, target: Optional[torch.Tensor] = None) -> SafetyCheckResult:
        """Validate that gradients flow properly through the model"""
        try:
            model.train()  # Set to training mode
            
            # Create a simple loss function if target not provided
            if target is None:
                target = torch.randn_like(test_input) if test_input.dim() > 1 else torch.randint(0, 2, test_input.shape[:1])
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            output = model(test_input)
            
            # Compute loss
            if target.dtype in [torch.long, torch.int64]:
                # Classification task
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            else:
                # Regression task
                criterion = nn.MSELoss()
                loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Check if gradients exist for parameters
            params_with_grad = 0
            total_params = 0
            zero_grad_params = 0
            
            for param in model.parameters():
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    if torch.allclose(param.grad, torch.zeros_like(param.grad)):
                        zero_grad_params += 1
            
            passed = params_with_grad > 0 and zero_grad_params / total_params < 0.5  # Less than 50% zero gradients
            
            message = f"Gradient flow: {params_with_grad}/{total_params} parameters have gradients, {zero_grad_params} have zero gradients. {'PASSED' if passed else 'FAILED'}"
            
            result = SafetyCheckResult(
                check_type=SafetyCheckType.GRADIENT_FLOW,
                passed=passed,
                message=message,
                details={
                    'params_with_grad': params_with_grad,
                    'total_params': total_params,
                    'zero_grad_params': zero_grad_params,
                    'grad_flow_percentage': params_with_grad / total_params if total_params > 0 else 0
                }
            )
            
            self.safety_check_results.append(result)
            return result
            
        except Exception as e:
            message = f"Gradient flow validation failed with error: {e}"
            result = SafetyCheckResult(
                check_type=SafetyCheckType.GRADIENT_FLOW,
                passed=False,
                message=message,
                details={'error': str(e)}
            )
            self.safety_check_results.append(result)
            return result
    
    def run_all_safety_checks(self, model: nn.Module, test_input: torch.Tensor, 
                            baseline_param_count: Optional[int] = None) -> List[SafetyCheckResult]:
        """Run all safety checks on the model"""
        self.logger.info("Running all safety checks for model capacity preservation...")
        
        # Clear previous results
        self.safety_check_results = []
        
        results = []
        
        # Run each safety check
        results.append(self.validate_layer_count(model))
        results.append(self.validate_attention_heads(model))
        results.append(self.validate_parameter_count(model, baseline_param_count))
        results.append(self.validate_model_architecture(model))
        results.append(self.validate_forward_pass_integrity(model, test_input))
        results.append(self.validate_gradient_flow(model, test_input))
        
        # Log summary
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        
        self.logger.info(f"Safety checks summary: {passed_count}/{total_count} passed")
        
        for result in results:
            status = "PASSED" if result.passed else "FAILED"
            self.logger.info(f"  {result.check_type.value}: {status}")
        
        return results
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate a safety validation report"""
        if not self.safety_check_results:
            return {"message": "No safety checks have been run yet"}
        
        report = {
            'timestamp': torch.tensor(time.time()).item(),  # Using torch for consistency
            'total_checks': len(self.safety_check_results),
            'passed_checks': sum(1 for r in self.safety_check_results if r.passed),
            'failed_checks': sum(1 for r in self.safety_check_results if not r.passed),
            'check_results': [
                {
                    'check_type': r.check_type.value,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.safety_check_results
            ],
            'model_capacity_preserved': all(r.passed for r in self.safety_check_results)
        }
        
        return report


class CapacityPreservationManager:
    """
    Manages capacity preservation across all optimization techniques.
    Provides hooks to validate capacity before and after optimizations.
    """
    
    def __init__(self):
        self.validator = ModelCapacityValidator()
        self.logger = logging.getLogger(__name__)
        self.capacity_preserved = True
        self.baseline_model_signature = None
        self.baseline_param_count = None
    
    def set_baseline(self, baseline_model: nn.Module):
        """Set the baseline model for comparison"""
        self.baseline_param_count = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
        self.baseline_model_signature = self.validator._get_model_signature(baseline_model)
        self.logger.info(f"Baseline set: {self.baseline_param_count:,} parameters")
    
    def validate_before_optimization(self, model: nn.Module, test_input: torch.Tensor) -> bool:
        """Validate model capacity before applying optimizations"""
        self.logger.info("Validating model capacity before optimization...")
        
        results = self.validator.run_all_safety_checks(model, test_input, self.baseline_param_count)
        
        # Check if capacity is preserved
        self.capacity_preserved = all(r.passed for r in results)
        
        if not self.capacity_preserved:
            failed_checks = [r for r in results if not r.passed]
            self.logger.warning(f"Capacity validation failed before optimization: {len(failed_checks)} checks failed")
            for check in failed_checks:
                self.logger.warning(f"  - {check.message}")
        
        return self.capacity_preserved
    
    def validate_after_optimization(self, model: nn.Module, test_input: torch.Tensor) -> bool:
        """Validate model capacity after applying optimizations"""
        self.logger.info("Validating model capacity after optimization...")
        
        results = self.validator.run_all_safety_checks(model, test_input, self.baseline_param_count)
        
        # Check if capacity is preserved
        current_preserved = all(r.passed for r in results)
        self.capacity_preserved = self.capacity_preserved and current_preserved
        
        if not current_preserved:
            failed_checks = [r for r in results if not r.passed]
            self.logger.warning(f"Capacity validation failed after optimization: {len(failed_checks)} checks failed")
            for check in failed_checks:
                self.logger.warning(f"  - {check.message}")
        
        return current_preserved
    
    def enforce_capacity_preservation(self, model: nn.Module, test_input: torch.Tensor) -> bool:
        """Enforce capacity preservation by validating and potentially reverting changes"""
        if not self.validate_after_optimization(model, test_input):
            self.logger.error("Model capacity not preserved! Optimization may have damaged the model structure.")
            # In a real implementation, we might revert to a previous state here
            return False
        return True
    
    def get_capacity_status(self) -> Dict[str, Any]:
        """Get current capacity preservation status"""
        return {
            'capacity_preserved': self.capacity_preserved,
            'baseline_parameters': self.baseline_param_count,
            'last_validation_results': self.validator.get_safety_report()
        }
    
    def register_capacity_hooks(self, model: nn.Module):
        """Register hooks to monitor capacity during training/optimization"""
        # Register forward hooks to monitor output shapes
        def output_shape_hook(module, input, output):
            if hasattr(output, 'shape'):
                self.logger.debug(f"Module {type(module).__name__} output shape: {output.shape}")
        
        # Register hooks on key modules
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.register_forward_hook(output_shape_hook)
        
        self.logger.info("Capacity monitoring hooks registered")


class OptimizationSafetyValidator:
    """
    Comprehensive safety validator that integrates with the optimization workflow
    to ensure capacity preservation throughout the optimization process.
    """
    
    def __init__(self, capacity_manager: CapacityPreservationManager):
        self.capacity_manager = capacity_manager
        self.logger = logging.getLogger(__name__)
    
    def validate_optimization_impact(self, 
                                   original_model: nn.Module, 
                                   optimized_model: nn.Module, 
                                   test_input: torch.Tensor) -> Dict[str, Any]:
        """Validate the impact of optimizations on model capacity"""
        self.logger.info("Validating optimization impact on model capacity...")
        
        # Validate original model
        original_valid = self.capacity_manager.validate_before_optimization(original_model, test_input)
        
        # Validate optimized model
        optimized_valid = self.capacity_manager.validate_after_optimization(optimized_model, test_input)
        
        # Compare architectures
        orig_sig = self.capacity_manager.validator._get_model_signature(original_model)
        opt_sig = self.capacity_manager.validator._get_model_signature(optimized_model)
        
        architecture_changed = orig_sig != opt_sig
        
        results = {
            'original_model_valid': original_valid,
            'optimized_model_valid': optimized_valid,
            'architecture_changed': architecture_changed,
            'capacity_preserved': original_valid and optimized_valid and not architecture_changed,
            'original_validation_report': self.capacity_manager.validator.get_safety_report(),
            'optimization_warnings': []
        }
        
        # Add warnings if architecture changed but model still functions
        if architecture_changed and optimized_valid:
            results['optimization_warnings'].append(
                "Model architecture changed during optimization but capacity preserved"
            )
        
        # Add warnings if capacity was affected
        if not optimized_valid:
            results['optimization_warnings'].append(
                "Model capacity validation failed after optimization"
            )
        
        return results
    
    def safe_optimize_model(self, 
                          model: nn.Module, 
                          optimization_func: callable, 
                          test_input: torch.Tensor,
                          *args, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
        """Safely apply optimizations with capacity preservation validation"""
        self.logger.info("Safely applying optimizations with capacity preservation...")
        
        # Store original model for comparison
        original_model = model  # In practice, you might want to deepcopy
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        try:
            # Validate before optimization
            if not self.capacity_manager.validate_before_optimization(model, test_input):
                raise ValueError("Original model failed capacity validation")
            
            # Apply optimization
            optimized_model = optimization_func(model, *args, **kwargs)
            
            # Validate after optimization
            if not self.capacity_manager.validate_after_optimization(optimized_model, test_input):
                self.logger.warning("Optimization compromised model capacity, reverting...")
                # Restore original model
                optimized_model.load_state_dict(original_state_dict)
                return optimized_model, {
                    'success': False,
                    'error': 'Capacity preservation failed',
                    'capacity_preserved': False
                }
            
            return optimized_model, {
                'success': True,
                'capacity_preserved': True,
                'validation_report': self.capacity_manager.get_capacity_status()
            }
            
        except Exception as e:
            self.logger.error(f"Safe optimization failed: {e}")
            # Restore original model
            model.load_state_dict(original_state_dict)
            return model, {
                'success': False,
                'error': str(e),
                'capacity_preserved': False
            }


# Global capacity preservation manager
def create_capacity_preservation_manager() -> CapacityPreservationManager:
    """Create a global capacity preservation manager instance"""
    return CapacityPreservationManager()