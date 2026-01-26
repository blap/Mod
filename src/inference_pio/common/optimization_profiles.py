"""
Optimization Profile System for Inference-PIO

This module provides a comprehensive system for defining, loading, and applying 
optimization profiles to different models in the Inference-PIO system.
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class OptimizationProfile:
    """
    Base class for optimization profiles that can be applied to different models.
    """

    def __init__(self, name: str, description: str = "", **kwargs):
        self.name = name
        self.description = description
        self.created_at = kwargs.get('created_at', '')
        self.version = kwargs.get('version', '1.0')
        self.tags = kwargs.get('tags', [])
        
        # Store additional profile-specific parameters
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary."""
        result = {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'version': self.version,
            'tags': self.tags
        }
        
        # Add any additional attributes
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in result:
                attr_value = getattr(self, attr_name)
                if not callable(attr_value):
                    result[attr_name] = attr_value
                    
        return result

    def apply_to_config(self, config) -> None:
        """
        Apply this profile to a model configuration.
        
        Args:
            config: Model configuration object to modify
        """
        profile_dict = self.to_dict()
        
        # Apply profile parameters to config
        for key, value in profile_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def clone(self):
        """Create a deep copy of the profile."""
        return copy.deepcopy(self)


@dataclass
class PerformanceProfile(OptimizationProfile):
    """
    Performance-focused optimization profile.
    """
    
    def __init__(self, name: str = "performance", description: str = "Performance-optimized settings", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # Performance-focused settings
        self.use_flash_attention_2: bool = kwargs.get('use_flash_attention_2', True)
        self.use_sparse_attention: bool = kwargs.get('use_sparse_attention', True)
        self.use_paged_attention: bool = kwargs.get('use_paged_attention', True)
        self.gradient_checkpointing: bool = kwargs.get('gradient_checkpointing', False)  # Disable for better performance
        self.torch_compile_mode: str = kwargs.get('torch_compile_mode', 'reduce-overhead')
        self.torch_compile_fullgraph: bool = kwargs.get('torch_compile_fullgraph', False)
        self.enable_kernel_fusion: bool = kwargs.get('enable_kernel_fusion', True)
        self.use_tensor_parallelism: bool = kwargs.get('use_tensor_parallelism', True)
        self.enable_adaptive_batching: bool = kwargs.get('enable_adaptive_batching', True)
        self.max_batch_size: int = kwargs.get('max_batch_size', 32)
        self.use_quantization: bool = kwargs.get('use_quantization', True)
        self.quantization_bits: int = kwargs.get('quantization_bits', 8)
        self.enable_memory_efficient_attention: bool = kwargs.get('enable_memory_efficient_attention', True)
        self.use_fused_layer_norm: bool = kwargs.get('use_fused_layer_norm', True)
        self.use_bias_removal_optimization: bool = kwargs.get('use_bias_removal_optimization', True)
        self.linear_bias_optimization_enabled: bool = kwargs.get('linear_bias_optimization_enabled', True)
        self.enable_cudnn_benchmark: bool = kwargs.get('enable_cudnn_benchmark', True)
        self.use_cuda_kernels: bool = kwargs.get('use_cuda_kernels', True)
        self.enable_sequence_parallelism: bool = kwargs.get('enable_sequence_parallelism', True)
        self.use_multi_query_attention: bool = kwargs.get('use_multi_query_attention', True)
        self.use_grouped_query_attention: bool = kwargs.get('use_grouped_query_attention', True)
        self.use_sliding_window_attention: bool = kwargs.get('use_sliding_window_attention', True)
        self.use_kv_cache_compression: bool = kwargs.get('use_kv_cache_compression', True)
        self.kv_cache_compression_method: str = kwargs.get('kv_cache_compression_method', "quantization")


@dataclass
class MemoryEfficientProfile(OptimizationProfile):
    """
    Memory-efficient optimization profile.
    """
    
    def __init__(self, name: str = "memory_efficient", description: str = "Memory-efficient settings", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # Memory-efficient settings
        self.use_flash_attention_2: bool = kwargs.get('use_flash_attention_2', True)
        self.use_sparse_attention: bool = kwargs.get('use_sparse_attention', True)
        self.use_paged_attention: bool = kwargs.get('use_paged_attention', True)
        self.gradient_checkpointing: bool = kwargs.get('gradient_checkpointing', True)  # Enable for memory savings
        self.torch_dtype: str = kwargs.get('torch_dtype', 'float16')  # Use half precision
        self.enable_memory_management: bool = kwargs.get('enable_memory_management', True)
        self.max_memory_ratio: float = kwargs.get('max_memory_ratio', 0.6)  # Limit memory usage
        self.enable_disk_offloading: bool = kwargs.get('enable_disk_offloading', True)
        self.enable_activation_offloading: bool = kwargs.get('enable_activation_offloading', True)
        self.enable_tensor_compression: bool = kwargs.get('enable_tensor_compression', True)
        self.tensor_compression_ratio: float = kwargs.get('tensor_compression_ratio', 0.5)
        self.use_tensor_decomposition: bool = kwargs.get('use_tensor_decomposition', True)
        self.use_structured_pruning: bool = kwargs.get('use_structured_pruning', True)
        self.pruning_ratio: float = kwargs.get('pruning_ratio', 0.2)
        self.use_quantization: bool = kwargs.get('use_quantization', True)
        self.quantization_bits: int = kwargs.get('quantization_bits', 4)
        self.enable_predictive_management: bool = kwargs.get('enable_predictive_management', True)
        self.enable_predictive_offloading: bool = kwargs.get('enable_predictive_offloading', True)
        self.enable_predictive_activation_offloading: bool = kwargs.get('enable_predictive_activation_offloading', True)
        self.enable_intelligent_pagination: bool = kwargs.get('enable_intelligent_pagination', True)
        self.pagination_max_memory_ratio: float = kwargs.get('pagination_max_memory_ratio', 0.6)
        self.enable_tensor_paging: bool = kwargs.get('enable_tensor_paging', True)
        self.page_size_mb: int = kwargs.get('page_size_mb', 8)


@dataclass
class BalancedProfile(OptimizationProfile):
    """
    Balanced optimization profile.
    """
    
    def __init__(self, name: str = "balanced", description: str = "Balanced performance and memory usage", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # Balanced settings
        self.use_flash_attention_2: bool = kwargs.get('use_flash_attention_2', True)
        self.use_sparse_attention: bool = kwargs.get('use_sparse_attention', True)
        self.use_paged_attention: bool = kwargs.get('use_paged_attention', True)
        self.gradient_checkpointing: bool = kwargs.get('gradient_checkpointing', True)
        self.torch_compile_mode: str = kwargs.get('torch_compile_mode', 'reduce-overhead')
        self.torch_compile_fullgraph: bool = kwargs.get('torch_compile_fullgraph', False)
        self.enable_kernel_fusion: bool = kwargs.get('enable_kernel_fusion', True)
        self.enable_memory_management: bool = kwargs.get('enable_memory_management', True)
        self.max_memory_ratio: float = kwargs.get('max_memory_ratio', 0.8)
        self.enable_disk_offloading: bool = kwargs.get('enable_disk_offloading', True)
        self.enable_activation_offloading: bool = kwargs.get('enable_activation_offloading', True)
        self.enable_tensor_compression: bool = kwargs.get('enable_tensor_compression', True)
        self.tensor_compression_ratio: float = kwargs.get('tensor_compression_ratio', 0.4)
        self.use_quantization: bool = kwargs.get('use_quantization', True)
        self.quantization_bits: int = kwargs.get('quantization_bits', 8)
        self.enable_adaptive_batching: bool = kwargs.get('enable_adaptive_batching', True)
        self.max_batch_size: int = kwargs.get('max_batch_size', 16)
        self.use_tensor_parallelism: bool = kwargs.get('use_tensor_parallelism', False)
        self.enable_predictive_management: bool = kwargs.get('enable_predictive_management', True)
        self.enable_intelligent_pagination: bool = kwargs.get('enable_intelligent_pagination', True)
        self.pagination_max_memory_ratio: float = kwargs.get('pagination_max_memory_ratio', 0.7)


class CustomProfile(OptimizationProfile):
    """
    Custom optimization profile allowing arbitrary settings.
    """

    def __init__(self, name: str, description: str = "Custom optimization profile", **kwargs):
        super().__init__(name, description, **kwargs)

        # Allow arbitrary parameters to be passed in
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class ProfileManager:
    """
    Centralized manager for handling optimization profiles across all models.
    """

    def __init__(self):
        self._profiles: Dict[str, OptimizationProfile] = {}
        self._profile_history: Dict[str, List[OptimizationProfile]] = {}
        self._active_profiles: Dict[str, str] = {}  # model_id -> profile_name mapping
        self._profile_templates: Dict[str, OptimizationProfile] = {}

    def register_profile_template(self, name: str, template: OptimizationProfile) -> bool:
        """
        Register a profile template that can be used to create new profiles.

        Args:
            name: Name of the template
            template: Template profile object

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._profile_templates[name] = template.clone()
            logger.info(f"Registered profile template: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register profile template {name}: {e}")
            return False

    def create_profile_from_template(self, template_name: str, profile_name: str,
                                   overrides: Optional[Dict[str, Any]] = None) -> Optional[OptimizationProfile]:
        """
        Create a new profile from a template with optional overrides.

        Args:
            template_name: Name of the template to use
            profile_name: Name for the new profile
            overrides: Optional dictionary of field overrides

        Returns:
            New profile object or None if creation failed
        """
        if template_name not in self._profile_templates:
            logger.error(f"Template '{template_name}' not found")
            return None

        try:
            # Clone the template
            new_profile = self._profile_templates[template_name].clone()

            # Update the profile's name and description to match the new profile
            new_profile.name = profile_name
            if overrides and 'description' in overrides:
                new_profile.description = overrides['description']

            # Apply overrides if provided
            if overrides:
                for key, value in overrides.items():
                    if hasattr(new_profile, key) and key not in ['name', 'description']:
                        setattr(new_profile, key, value)

            # Store the profile
            self._profiles[profile_name] = new_profile

            # Add to history
            if profile_name not in self._profile_history:
                self._profile_history[profile_name] = []
            self._profile_history[profile_name].append(new_profile.clone())

            logger.info(f"Created profile '{profile_name}' from template '{template_name}'")
            return new_profile
        except Exception as e:
            logger.error(f"Failed to create profile from template {template_name}: {e}")
            return None

    def register_profile(self, name: str, profile: OptimizationProfile) -> bool:
        """
        Register a profile with the manager.

        Args:
            name: Name of the profile
            profile: Profile object

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            self._profiles[name] = profile.clone()

            # Add to history
            if name not in self._profile_history:
                self._profile_history[name] = []
            self._profile_history[name].append(profile.clone())

            logger.info(f"Registered profile: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register profile {name}: {e}")
            return False

    def get_profile(self, name: str) -> Optional[OptimizationProfile]:
        """
        Get a profile by name.

        Args:
            name: Name of the profile

        Returns:
            Profile object or None if not found
        """
        return self._profiles.get(name)

    def update_profile(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update a profile with new values.

        Args:
            name: Name of the profile to update
            updates: Dictionary of field updates

        Returns:
            True if update was successful, False otherwise
        """
        if name not in self._profiles:
            logger.error(f"Profile '{name}' not found")
            return False

        try:
            profile = self._profiles[name]
            
            # Update profile attributes
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            # Add to history
            if name not in self._profile_history:
                self._profile_history[name] = []
            self._profile_history[name].append(profile.clone())

            logger.info(f"Updated profile: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update profile {name}: {e}")
            return False

    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile.

        Args:
            name: Name of the profile to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if name not in self._profiles:
            logger.error(f"Profile '{name}' not found")
            return False

        try:
            del self._profiles[name]
            if name in self._profile_history:
                del self._profile_history[name]

            # Remove from active profiles if present
            keys_to_remove = []
            for model_id, profile_name in self._active_profiles.items():
                if profile_name == name:
                    keys_to_remove.append(model_id)
            for key in keys_to_remove:
                del self._active_profiles[key]

            logger.info(f"Deleted profile: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete profile {name}: {e}")
            return False

    def save_profile(self, name: str, filepath: str, format: str = "json") -> bool:
        """
        Save a profile to a file.

        Args:
            name: Name of the profile to save
            filepath: Path to save the file
            format: Format to save in ("json" or "yaml")

        Returns:
            True if save was successful, False otherwise
        """
        profile = self.get_profile(name)
        if not profile:
            logger.error(f"Profile '{name}' not found")
            return False

        try:
            data = profile.to_dict()

            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved profile '{name}' to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile {name} to {filepath}: {e}")
            return False

    def load_profile(self, name: str, filepath: str, format: str = "json") -> bool:
        """
        Load a profile from a file.

        Args:
            name: Name to assign to the loaded profile
            filepath: Path to load the file from
            format: Format to load from ("json" or "yaml")

        Returns:
            True if load was successful, False otherwise
        """
        try:
            if format.lower() == "json":
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format.lower() == "yaml":
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Extract description from data if present, otherwise use default
            profile_description = data.pop('description', 'Loaded profile')
            # Remove name from data to avoid conflict with the name parameter
            data.pop('name', None)

            # Create a custom profile from the loaded data using the provided name
            profile = CustomProfile(name=name, description=profile_description, **data)
            return self.register_profile(name, profile)
        except Exception as e:
            logger.error(f"Failed to load profile {name} from {filepath}: {e}")
            return False

    def list_profiles(self) -> List[str]:
        """
        List all registered profiles.

        Returns:
            List of profile names
        """
        return list(self._profiles.keys())

    def get_profile_history(self, name: str) -> List[OptimizationProfile]:
        """
        Get the history of changes for a profile.

        Args:
            name: Name of the profile

        Returns:
            List of previous versions of the profile
        """
        return self._profile_history.get(name, [])

    def activate_profile_for_model(self, model_id: str, profile_name: str) -> bool:
        """
        Activate a profile for a specific model.

        Args:
            model_id: Identifier for the model
            profile_name: Name of the profile to activate

        Returns:
            True if activation was successful, False otherwise
        """
        if profile_name not in self._profiles:
            logger.error(f"Profile '{profile_name}' not found")
            return False

        self._active_profiles[model_id] = profile_name
        logger.info(f"Activated profile '{profile_name}' for model '{model_id}'")
        return True

    def get_active_profile_for_model(self, model_id: str) -> Optional[OptimizationProfile]:
        """
        Get the active profile for a specific model.

        Args:
            model_id: Identifier for the model

        Returns:
            Active profile or None if no profile is active
        """
        profile_name = self._active_profiles.get(model_id)
        if profile_name:
            return self.get_profile(profile_name)
        return None

    def get_profile_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a profile.

        Args:
            name: Name of the profile

        Returns:
            Metadata dictionary or None if profile not found
        """
        profile = self.get_profile(name)
        if not profile:
            return None

        return {
            "name": name,
            "type": type(profile).__name__,
            "description": profile.description,
            "version": profile.version,
            "tags": profile.tags,
            "history_length": len(self.get_profile_history(name)),
            "is_active_in_models": [model_id for model_id, profile_name in self._active_profiles.items()
                                   if profile_name == name]
        }

    def apply_profile_to_config(self, profile_name: str, config) -> bool:
        """
        Apply a profile to a model configuration.

        Args:
            profile_name: Name of the profile to apply
            config: Model configuration to modify

        Returns:
            True if application was successful, False otherwise
        """
        profile = self.get_profile(profile_name)
        if not profile:
            logger.error(f"Profile '{profile_name}' not found")
            return False

        try:
            profile.apply_to_config(config)
            logger.info(f"Applied profile '{profile_name}' to configuration")
            return True
        except Exception as e:
            logger.error(f"Failed to apply profile {profile_name} to config: {e}")
            return False


# Global profile manager instance
profile_manager = ProfileManager()


def get_profile_manager() -> ProfileManager:
    """
    Get the global profile manager instance.

    Returns:
        ProfileManager instance
    """
    return profile_manager


# Model-specific profile classes
class GLM47Profile(OptimizationProfile):
    """
    GLM-4.7 specific optimization profile.
    """
    
    def __init__(self, name: str = "glm_4_7_flash_default", description: str = "GLM-4.7-Flash default optimization profile", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # GLM-4.7 specific optimizations
        self.use_glm_attention_patterns: bool = kwargs.get('use_glm_attention_patterns', True)
        self.glm_attention_pattern_sparsity: float = kwargs.get('glm_attention_pattern_sparsity', 0.3)
        self.glm_attention_window_size: int = kwargs.get('glm_attention_window_size', 1024)
        self.use_glm_ffn_optimization: bool = kwargs.get('use_glm_ffn_optimization', True)
        self.glm_ffn_expansion_ratio: float = kwargs.get('glm_ffn_expansion_ratio', 2.6)
        self.glm_ffn_group_size: int = kwargs.get('glm_ffn_group_size', 128)
        self.use_glm_memory_efficient_kv: bool = kwargs.get('use_glm_memory_efficient_kv', True)
        self.glm_kv_cache_compression_ratio: float = kwargs.get('glm_kv_cache_compression_ratio', 0.5)
        self.use_glm_layer_norm_fusion: bool = kwargs.get('use_glm_layer_norm_fusion', True)
        self.use_glm_residual_connection_optimization: bool = kwargs.get('use_glm_residual_connection_optimization', True)
        self.use_glm_quantization: bool = kwargs.get('use_glm_quantization', True)
        self.glm_weight_bits: int = kwargs.get('glm_weight_bits', 4)
        self.glm_activation_bits: int = kwargs.get('glm_activation_bits', 8)


class Qwen34BProfile(OptimizationProfile):
    """
    Qwen3-4B-Instruct-2507 specific optimization profile.
    """
    
    def __init__(self, name: str = "qwen3_4b_default", description: str = "Qwen3-4B default optimization profile", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # Qwen3-4B specific optimizations
        self.use_qwen3_attention_optimizations: bool = kwargs.get('use_qwen3_attention_optimizations', True)
        self.use_qwen3_kv_cache_optimizations: bool = kwargs.get('use_qwen3_kv_cache_optimizations', True)
        self.use_qwen3_instruction_optimizations: bool = kwargs.get('use_qwen3_instruction_optimizations', True)
        self.use_qwen3_rope_optimizations: bool = kwargs.get('use_qwen3_rope_optimizations', True)
        self.use_qwen3_gqa_optimizations: bool = kwargs.get('use_qwen3_gqa_optimizations', True)
        self.qwen3_attention_sparsity_ratio: float = kwargs.get('qwen3_attention_sparsity_ratio', 0.3)
        self.qwen3_kv_cache_compression_ratio: float = kwargs.get('qwen3_kv_cache_compression_ratio', 0.6)
        self.qwen3_instruction_attention_scaling: float = kwargs.get('qwen3_instruction_attention_scaling', 1.2)
        self.qwen3_extended_context_optimization: bool = kwargs.get('qwen3_extended_context_optimization', True)
        self.qwen3_speculative_decoding_enabled: bool = kwargs.get('qwen3_speculative_decoding_enabled', False)
        self.qwen3_speculative_draft_model_ratio: float = kwargs.get('qwen3_speculative_draft_model_ratio', 0.5)
        self.qwen3_speculative_max_tokens: int = kwargs.get('qwen3_speculative_max_tokens', 5)
        self.qwen3_instruction_prompt_enhancement: bool = kwargs.get('qwen3_instruction_prompt_enhancement', True)
        self.qwen3_response_quality_optimization: bool = kwargs.get('qwen3_response_quality_optimization', True)
        self.qwen3_memory_efficient_inference: bool = kwargs.get('qwen3_memory_efficient_inference', True)
        self.qwen3_compute_efficient_inference: bool = kwargs.get('qwen3_compute_efficient_inference', True)


class Qwen3CoderProfile(OptimizationProfile):
    """
    Qwen3-Coder-30B specific optimization profile.
    """
    
    def __init__(self, name: str = "qwen3_coder_default", description: str = "Qwen3-Coder default optimization profile", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # Qwen3-Coder specific optimizations
        self.use_qwen3_coder_attention_optimizations: bool = kwargs.get('use_qwen3_coder_attention_optimizations', True)
        self.use_qwen3_coder_kv_cache_optimizations: bool = kwargs.get('use_qwen3_coder_kv_cache_optimizations', True)
        self.use_qwen3_coder_code_optimizations: bool = kwargs.get('use_qwen3_coder_code_optimizations', True)
        self.use_qwen3_coder_syntax_highlighting: bool = kwargs.get('use_qwen3_coder_syntax_highlighting', True)
        self.qwen3_coder_attention_sparsity_ratio: float = kwargs.get('qwen3_coder_attention_sparsity_ratio', 0.3)
        self.qwen3_coder_kv_cache_compression_ratio: float = kwargs.get('qwen3_coder_kv_cache_compression_ratio', 0.6)
        self.qwen3_coder_syntax_attention_scaling: float = kwargs.get('qwen3_coder_syntax_attention_scaling', 1.2)
        self.qwen3_coder_extended_context_optimization: bool = kwargs.get('qwen3_coder_extended_context_optimization', True)
        self.qwen3_coder_speculative_decoding_enabled: bool = kwargs.get('qwen3_coder_speculative_decoding_enabled', False)
        self.qwen3_coder_speculative_draft_model_ratio: float = kwargs.get('qwen3_coder_speculative_draft_model_ratio', 0.5)
        self.qwen3_coder_speculative_max_tokens: int = kwargs.get('qwen3_coder_speculative_max_tokens', 5)
        self.qwen3_coder_code_prompt_enhancement: bool = kwargs.get('qwen3_coder_code_prompt_enhancement', True)
        self.qwen3_coder_code_quality_optimization: bool = kwargs.get('qwen3_coder_code_quality_optimization', True)
        self.qwen3_coder_memory_efficient_inference: bool = kwargs.get('qwen3_coder_memory_efficient_inference', True)
        self.qwen3_coder_compute_efficient_inference: bool = kwargs.get('qwen3_coder_compute_efficient_inference', True)
        
        # Code-specific optimizations
        self.code_generation_temperature: float = kwargs.get('code_generation_temperature', 0.2)
        self.code_completion_top_p: float = kwargs.get('code_completion_top_p', 0.95)
        self.code_context_window_extension: int = kwargs.get('code_context_window_extension', 16384)
        self.code_special_tokens_handling: bool = kwargs.get('code_special_tokens_handling', True)
        self.code_syntax_aware_attention: bool = kwargs.get('code_syntax_aware_attention', True)
        self.code_identifiers_extraction: bool = kwargs.get('code_identifiers_extraction', True)
        self.code_syntax_validation: bool = kwargs.get('code_syntax_validation', True)
        self.code_comment_generation: bool = kwargs.get('code_comment_generation', True)
        self.code_refactoring_support: bool = kwargs.get('code_refactoring_support', True)
        self.code_error_correction: bool = kwargs.get('code_error_correction', True)
        self.code_style_consistency: bool = kwargs.get('code_style_consistency', True)
        self.code_library_detection: bool = kwargs.get('code_library_detection', True)
        self.code_security_scanning: bool = kwargs.get('code_security_scanning', True)
        self.code_complexity_optimization: bool = kwargs.get('code_complexity_optimization', True)


class Qwen3VLProfile(OptimizationProfile):
    """
    Qwen3-VL-2B specific optimization profile.
    """
    
    def __init__(self, name: str = "qwen3_vl_default", description: str = "Qwen3-VL default optimization profile", **kwargs):
        super().__init__(name, description, **kwargs)
        
        # Qwen3-VL specific optimizations
        self.use_qwen3_vl_attention_optimizations: bool = kwargs.get('use_qwen3_vl_attention_optimizations', True)
        self.use_qwen3_vl_kv_cache_optimizations: bool = kwargs.get('use_qwen3_vl_kv_cache_optimizations', True)
        self.use_qwen3_vl_vision_optimizations: bool = kwargs.get('use_qwen3_vl_vision_optimizations', True)
        self.use_qwen3_vl_cross_modal_optimizations: bool = kwargs.get('use_qwen3_vl_cross_modal_optimizations', True)
        self.qwen3_vl_attention_sparsity_ratio: float = kwargs.get('qwen3_vl_attention_sparsity_ratio', 0.3)
        self.qwen3_vl_kv_cache_compression_ratio: float = kwargs.get('qwen3_vl_kv_cache_compression_ratio', 0.6)
        self.qwen3_vl_cross_modal_attention_scaling: float = kwargs.get('qwen3_vl_cross_modal_attention_scaling', 1.2)
        self.qwen3_vl_extended_context_optimization: bool = kwargs.get('qwen3_vl_extended_context_optimization', True)
        self.qwen3_vl_speculative_decoding_enabled: bool = kwargs.get('qwen3_vl_speculative_decoding_enabled', False)
        self.qwen3_vl_speculative_draft_model_ratio: float = kwargs.get('qwen3_vl_speculative_draft_model_ratio', 0.5)
        self.qwen3_vl_speculative_max_tokens: int = kwargs.get('qwen3_vl_speculative_max_tokens', 5)
        self.qwen3_vl_vision_prompt_enhancement: bool = kwargs.get('qwen3_vl_vision_prompt_enhancement', True)
        self.qwen3_vl_vision_quality_optimization: bool = kwargs.get('qwen3_vl_vision_quality_optimization', True)
        self.qwen3_vl_memory_efficient_inference: bool = kwargs.get('qwen3_vl_memory_efficient_inference', True)
        self.qwen3_vl_compute_efficient_inference: bool = kwargs.get('qwen3_vl_compute_efficient_inference', True)
        
        # Vision-specific optimizations
        self.use_multimodal_attention: bool = kwargs.get('use_multimodal_attention', True)
        self.multimodal_attention_sparsity_ratio: float = kwargs.get('multimodal_attention_sparsity_ratio', 0.3)
        self.multimodal_attention_local_window_size: int = kwargs.get('multimodal_attention_local_window_size', 128)
        self.use_cross_modal_fusion: bool = kwargs.get('use_cross_modal_fusion', True)
        self.cross_modal_fusion_method: str = kwargs.get('cross_modal_fusion_method', "qwen3_vl_specific")
        self.use_cross_modal_alignment: bool = kwargs.get('use_cross_modal_alignment', True)
        self.cross_modal_alignment_method: str = kwargs.get('cross_modal_alignment_method', 'qwen3_vl_specific')
        self.enable_intelligent_multimodal_caching: bool = kwargs.get('enable_intelligent_multimodal_caching', True)
        self.intelligent_multimodal_cache_size_gb: float = kwargs.get('intelligent_multimodal_cache_size_gb', 2.0)
        self.enable_async_multimodal_processing: bool = kwargs.get('enable_async_multimodal_processing', True)
        self.enable_dynamic_multimodal_batching: bool = kwargs.get('enable_dynamic_multimodal_batching', False)
        self.enable_multimodal_preprocessing_pipeline: bool = kwargs.get('enable_multimodal_preprocessing_pipeline', True)
        self.enable_vision_attention_optimization: bool = kwargs.get('enable_vision_attention_optimization', True)
        self.enable_vision_mlp_optimization: bool = kwargs.get('enable_vision_mlp_optimization', True)
        self.enable_vision_block_optimization: bool = kwargs.get('enable_vision_block_optimization', True)
        self.enable_visual_resource_compression: bool = kwargs.get('enable_visual_resource_compression', True)
        self.visual_compression_method: str = kwargs.get('visual_compression_method', 'quantization')
        self.visual_compression_ratio: float = kwargs.get('visual_compression_ratio', 0.5)
        self.enable_image_tokenization: bool = kwargs.get('enable_image_tokenization', True)
        self.max_image_tokens: int = kwargs.get('max_image_tokens', 1024)


# Register default profile templates
def register_default_profile_templates():
    """Register default profile templates."""
    manager = get_profile_manager()

    # Register basic profile templates
    performance_template = PerformanceProfile()
    manager.register_profile_template("performance", performance_template)

    memory_efficient_template = MemoryEfficientProfile()
    manager.register_profile_template("memory_efficient", memory_efficient_template)

    balanced_template = BalancedProfile()
    manager.register_profile_template("balanced", balanced_template)

    # Register model-specific templates
    glm_template = GLM47Profile()
    manager.register_profile_template("glm_4_7_flash", glm_template)

    qwen3_4b_template = Qwen34BProfile()
    manager.register_profile_template("qwen3_4b_instruct_2507", qwen3_4b_template)

    qwen3_coder_template = Qwen3CoderProfile()
    manager.register_profile_template("qwen3_coder_30b", qwen3_coder_template)

    qwen3_vl_template = Qwen3VLProfile()
    manager.register_profile_template("qwen3_vl_2b", qwen3_vl_template)


# Register templates on module import
register_default_profile_templates()


__all__ = [
    "OptimizationProfile",
    "PerformanceProfile",
    "MemoryEfficientProfile",
    "BalancedProfile",
    "CustomProfile",
    "ProfileManager",
    "get_profile_manager",
    "profile_manager",
    "GLM47Profile",
    "Qwen34BProfile",
    "Qwen3CoderProfile",
    "Qwen3VLProfile",
    "register_default_profile_templates",
]