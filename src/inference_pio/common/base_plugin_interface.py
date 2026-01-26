"""
Base Plugin Interface for Inference-PIO System

This module defines the base interfaces for plugins in the Inference-PIO system.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, Tuple
from enum import Enum

import torch
import torch.nn as nn
from .model_surgery import ModelSurgerySystem, apply_model_surgery, restore_model_from_surgery
from .activation_offloading import ActivationOffloadingManager, ActivationPriority, ActivationAccessPattern
from .security_manager import SecurityLevel, ResourceLimits, initialize_plugin_isolation, cleanup_plugin_isolation


# Import the standardized interface components
from .standard_plugin_interface import (
    PluginType,
    PluginMetadata as ModelPluginMetadata,
    StandardPluginInterface
)


class ModelPluginInterface(StandardPluginInterface):
    """
    Base interface for all model plugins in the Inference-PIO system.
    """

    def __init__(self, metadata: ModelPluginMetadata):
        super().__init__(metadata)
        self._compiled_model = None
        # Sharding and streaming attributes
        self._sharder = None
        self._streaming_loader = None
        self._sharding_enabled = False
        self._current_inference_context = None

        # Activation offloading attributes
        self._activation_offloading_manager = None
        self._activation_offloading_enabled = False
        self._activation_offloading_config = {}

        # Security and isolation attributes
        self._security_initialized = False
        self._security_level = SecurityLevel.MEDIUM_TRUST
        self._resource_limits = None

    # Methods inherited from StandardPluginInterface:
    # - initialize (abstract)
    # - load_model (abstract)
    # - infer (abstract)
    # - cleanup (abstract)
    # - supports_config (abstract)

    @abstractmethod
    def supports_config(self, config: Any) -> bool:
        """
        Check if this plugin supports the given configuration.

        Args:
            config: Configuration to check

        Returns:
            True if the configuration is supported, False otherwise
        """

    def enable_sharding(self, num_shards: int = 500, storage_path: str = "./shards", **kwargs) -> bool:
        """
        Enable extreme sharding for the model.

        Args:
            num_shards: Number of shards to create (default 500 for extreme sharding)
            storage_path: Path to store shard files
            **kwargs: Additional sharding configuration parameters

        Returns:
            True if sharding was enabled successfully, False otherwise
        """
        try:
            from .model_sharder import create_extreme_sharding_system
            self._sharder, self._streaming_loader = create_extreme_sharding_system(
                storage_path=storage_path,
                num_shards=num_shards
            )
            self._sharding_enabled = True
            logger.info(f"Extreme sharding enabled with {num_shards} shards")
            return True
        except Exception as e:
            logger.error(f"Failed to enable sharding: {e}")
            return False

    def disable_sharding(self) -> bool:
        """
        Disable sharding for the model.

        Returns:
            True if sharding was disabled successfully, False otherwise
        """
        try:
            if self._sharder:
                self._sharder.cleanup()
                self._sharder = None
            if self._streaming_loader:
                self._streaming_loader.cleanup_all_contexts()
                self._streaming_loader = None
            self._sharding_enabled = False
            logger.info("Sharding disabled")
            return True
        except Exception as e:
            logger.error(f"Failed to disable sharding: {e}")
            return False

    def shard_model(self, model: nn.Module, num_shards: int = 500) -> bool:
        """
        Shard the model into hundreds of tiny fragments.

        Args:
            model: Model to shard
            num_shards: Number of shards to create

        Returns:
            True if sharding was successful, False otherwise
        """
        try:
            if not self._sharder:
                logger.error("Sharding not enabled, call enable_sharding first")
                return False

            self._sharder.shard_model(model, num_shards)
            logger.info(f"Model successfully sharded into {num_shards} fragments")
            return True
        except Exception as e:
            logger.error(f"Failed to shard model: {e}")
            return False

    def prepare_inference_context(self, context_id: str, input_shape: Tuple,
                                inference_type: str = "forward") -> List[str]:
        """
        Prepare an inference context by determining and loading required shards.

        Args:
            context_id: Unique identifier for this inference context
            input_shape: Shape of the input tensor
            inference_type: Type of inference ("forward", "generate", etc.)

        Returns:
            List of shard IDs loaded for this context
        """
        try:
            if not self._streaming_loader:
                logger.error("Streaming loader not initialized")
                return []

            return self._streaming_loader.prepare_inference_context(
                context_id, input_shape, inference_type
            )
        except Exception as e:
            logger.error(f"Failed to prepare inference context: {e}")
            return []

    def execute_with_shards(self, context_id: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute inference in a prepared context using only required shards.

        Args:
            context_id: Context ID from prepare_inference_context
            input_tensor: Input tensor for inference

        Returns:
            Output tensor from the computation
        """
        try:
            if not self._streaming_loader:
                logger.error("Streaming loader not initialized")
                return input_tensor

            return self._streaming_loader.execute_in_context(context_id, input_tensor)
        except Exception as e:
            logger.error(f"Failed to execute with shards: {e}")
            # Fallback to regular inference if sharding fails
            return self.infer(input_tensor)

    def cleanup_inference_context(self, context_id: str, force_unload: bool = True):
        """
        Clean up an inference context and optionally unload shards.

        Args:
            context_id: Context ID to clean up
            force_unload: Whether to force unload all shards for this context
        """
        try:
            if self._streaming_loader:
                self._streaming_loader.cleanup_context(context_id, force_unload)
        except Exception as e:
            logger.error(f"Failed to cleanup inference context: {e}")

    def get_sharding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sharding system.

        Returns:
            Dictionary containing sharding statistics
        """
        try:
            if self._sharder:
                return self._sharder.get_memory_usage()
            else:
                return {
                    'sharding_enabled': False,
                    'total_shards': 0,
                    'loaded_shards': 0,
                    'total_size_bytes': 0,
                    'loaded_size_bytes': 0,
                    'memory_utilization_ratio': 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get sharding stats: {e}")
            return {}

    def setup_memory_management(self, **kwargs) -> bool:
        """
        Set up memory management including swap and paging configurations.

        Args:
            **kwargs: Memory management configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_tensor_paging(self, **kwargs) -> bool:
        """
        Enable tensor paging for the model to move parts between RAM and disk.

        Args:
            **kwargs: Tensor paging configuration parameters

        Returns:
            True if tensor paging was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_smart_swap(self, **kwargs) -> bool:
        """
        Enable smart swap functionality to configure additional swap on OS level.

        Args:
            **kwargs: Smart swap configuration parameters

        Returns:
            True if smart swap was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for the plugin.

        Returns:
            Dictionary containing memory statistics
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def force_memory_cleanup(self) -> bool:
        """
        Force cleanup of memory resources including cached tensors and swap files.

        Returns:
            True if cleanup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def start_predictive_memory_management(self, **kwargs) -> bool:
        """
        Start predictive memory management using ML algorithms to anticipate memory needs.

        Args:
            **kwargs: Configuration parameters for predictive management

        Returns:
            True if predictive management was started successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def stop_predictive_memory_management(self) -> bool:
        """
        Stop predictive memory management.

        Returns:
            True if predictive management was stopped successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def optimize_model(self, model: nn.Module = None, mode: str = 'reduce-overhead', fullgraph: bool = False, dynamic: bool = True) -> bool:
        """
        Apply runtime memory optimization using torch.compile.

        Args:
            model: Model to optimize (if None, uses self._model if available)
            mode: Compilation mode ('reduce-overhead', 'max-autotune', etc.)
            fullgraph: Whether to compile the entire forward pass as a single graph
            dynamic: Whether to enable dynamic shape compilation

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Determine which model to optimize
            target_model = model
            if target_model is None:
                # Try to get the model from the plugin instance (subclasses should set this)
                if hasattr(self, '_model') and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found, cannot optimize")
                    return False

            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True

            # Compile the model with specified optimizations
            self._compiled_model = torch.compile(
                target_model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )

            logger.info(f"Model optimized with torch.compile using mode: {mode}")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return False

    def get_compiled_model(self):
        """
        Get the compiled model if available, otherwise return the original model.

        Returns:
            Compiled model if available, otherwise original model
        """
        return self._compiled_model if self._compiled_model is not None else self._model

    def clear_cuda_cache(self) -> bool:
        """
        Clear CUDA cache to free up memory.

        Returns:
            True if cache was cleared successfully, False otherwise
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("CUDA cache cleared successfully")
                return True
            else:
                logger.info("CUDA not available, skipping cache clearing")
                return True
        except Exception as e:
            logger.error(f"Failed to clear CUDA cache: {e}")
            return False

    def setup_kernel_fusion(self, **kwargs) -> bool:
        """
        Set up kernel fusion system for optimizing model operations.

        Args:
            **kwargs: Kernel fusion configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def setup_disk_offloading(self, **kwargs) -> bool:
        """
        Set up disk offloading system for managing model components between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_disk_offloading(self, **kwargs) -> bool:
        """
        Enable disk offloading for the model to move parts between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if disk offloading was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def offload_model_parts(self, **kwargs) -> bool:
        """
        Offload specific model parts to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def predict_model_part_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which model parts will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping model part names to access probabilities
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def get_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about disk offloading operations.

        Returns:
            Dictionary containing offloading statistics
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def apply_kernel_fusion(self, model: nn.Module = None) -> bool:
        """
        Apply kernel fusion optimizations to the model.

        Args:
            model: Model to optimize (if None, uses self._model if available)

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Determine which model to optimize
            target_model = model
            if target_model is None:
                # Try to get the model from the plugin instance (subclasses should set this)
                if hasattr(self, '_model') and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found, cannot apply kernel fusion")
                    return False

            # Import kernel fusion manager
            from .kernel_fusion import get_kernel_fusion_manager
            fusion_manager = get_kernel_fusion_manager()

            # Apply kernel fusion optimizations
            self._model = fusion_manager.optimize_model(target_model)

            logger.info("Kernel fusion optimizations applied successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to apply kernel fusion: {e}")
            return False

    def get_fusion_manager(self):
        """
        Get the kernel fusion manager instance.

        Returns:
            Kernel fusion manager instance
        """
        try:
            from .kernel_fusion import get_kernel_fusion_manager
            return get_kernel_fusion_manager()
        except ImportError:
            logger.warning("Kernel fusion module not available")
            return None

    def setup_adaptive_batching(self, **kwargs) -> bool:
        """
        Set up adaptive batching system for dynamic batch size adjustment.

        Args:
            **kwargs: Adaptive batching configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def get_optimal_batch_size(self, processing_time_ms: float, tokens_processed: int) -> int:
        """
        Get the optimal batch size based on current performance metrics.

        Args:
            processing_time_ms: Processing time for the current batch in milliseconds
            tokens_processed: Number of tokens processed in the current batch

        Returns:
            Recommended batch size for the next batch
        """
        # Default implementation - return a fixed batch size
        return 1

    def adjust_batch_size(self) -> Tuple[int, bool, Optional[str]]:
        """
        Adjust the batch size based on current metrics.

        Returns:
            Tuple of (new_batch_size, was_adjusted, reason_for_adjustment)
        """
        # Default implementation - return current batch size without adjustment
        return 1, False, None

    def get_batching_status(self) -> Dict[str, Any]:
        """
        Get the current status of the adaptive batching system.

        Returns:
            Dictionary containing batching status information
        """
        # Default implementation - return basic status
        return {
            'current_batch_size': 1,
            'adaptive_batching_enabled': False,
            'memory_pressure_ratio': 0.0,
            'performance_score': 0.0
        }

    def setup_model_surgery(self, **kwargs) -> bool:
        """
        Set up model surgery system for identifying and removing non-essential components.

        Args:
            **kwargs: Model surgery configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Initialize model surgery system
            self._model_surgery_system = ModelSurgerySystem()

            # Store surgery configuration
            self._surgery_config = {
                'enabled': kwargs.get('surgery_enabled', True),
                'auto_identify': kwargs.get('auto_identify_components', True),
                'preserve_components': kwargs.get('preserve_components', []),
                'surgery_priority_threshold': kwargs.get('surgery_priority_threshold', 10),
                'analysis_only': kwargs.get('analysis_only', False)
            }

            logger.info("Model surgery system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup model surgery: {e}")
            return False

    def enable_model_surgery(self, **kwargs) -> bool:
        """
        Enable model surgery for the plugin to identify and temporarily remove
        non-essential components during inference.

        Args:
            **kwargs: Model surgery configuration parameters

        Returns:
            True if model surgery was enabled successfully, False otherwise
        """
        try:
            if not hasattr(self, '_model_surgery_system'):
                if not self.setup_model_surgery(**kwargs):
                    logger.error("Failed to setup model surgery system")
                    return False

            # Update surgery configuration with new parameters
            for key, value in kwargs.items():
                if hasattr(self, '_surgery_config'):
                    self._surgery_config[key] = value

            logger.info("Model surgery enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable model surgery: {e}")
            return False

    def perform_model_surgery(self, model: nn.Module = None,
                            components_to_remove: Optional[List[str]] = None,
                            preserve_components: Optional[List[str]] = None) -> nn.Module:
        """
        Perform model surgery by identifying and removing non-essential components.

        Args:
            model: Model to perform surgery on (if None, uses self._model)
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable

        Returns:
            Modified model with surgery applied
        """
        try:
            if not hasattr(self, '_model_surgery_system'):
                logger.error("Model surgery system not initialized")
                return model or self._model

            # Determine which model to operate on
            target_model = model
            if target_model is None:
                if hasattr(self, '_model') and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found")
                    return model

            # Use config values or fallback to parameters
            if preserve_components is None:
                preserve_components = self._surgery_config.get('preserve_components', [])

            # Perform the surgery
            self._model = apply_model_surgery(
                target_model,
                components_to_remove=components_to_remove,
                preserve_components=preserve_components
            )

            logger.info(f"Model surgery performed successfully")
            return self._model
        except Exception as e:
            logger.error(f"Failed to perform model surgery: {e}")
            # Return original model on failure
            return model or self._model

    def restore_model_from_surgery(self, model: nn.Module = None,
                                 surgery_id: Optional[str] = None) -> nn.Module:
        """
        Restore a model from surgery by putting back removed components.

        Args:
            model: Model to restore (if None, uses self._model)
            surgery_id: Specific surgery to reverse (None means reverse latest)

        Returns:
            Restored model
        """
        try:
            if not hasattr(self, '_model_surgery_system'):
                logger.error("Model surgery system not initialized")
                return model or self._model

            # Determine which model to operate on
            target_model = model
            if target_model is None:
                if hasattr(self, '_model') and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found")
                    return model

            # Perform restoration
            self._model = restore_model_from_surgery(target_model, surgery_id)

            logger.info(f"Model restoration from surgery completed successfully")
            return self._model
        except Exception as e:
            logger.error(f"Failed to restore model from surgery: {e}")
            # Return original model on failure
            return model or self._model

    def analyze_model_for_surgery(self, model: nn.Module = None) -> Dict[str, Any]:
        """
        Analyze a model to identify potential candidates for surgical removal.

        Args:
            model: Model to analyze (if None, uses self._model)

        Returns:
            Dictionary containing analysis results
        """
        try:
            if not hasattr(self, '_model_surgery_system'):
                logger.error("Model surgery system not initialized")
                return {}

            # Determine which model to operate on
            target_model = model
            if target_model is None:
                if hasattr(self, '_model') and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found")
                    return {}

            # Perform analysis
            analysis = self._model_surgery_system.analyze_model_for_surgery(target_model)

            logger.info(f"Model analysis for surgery completed")
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze model for surgery: {e}")
            return {}

    def get_surgery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about performed model surgeries.

        Returns:
            Dictionary containing surgery statistics
        """
        try:
            if not hasattr(self, '_model_surgery_system'):
                logger.error("Model surgery system not initialized")
                return {}

            return self._model_surgery_system.get_surgery_stats()
        except Exception as e:
            logger.error(f"Failed to get surgery stats: {e}")
            return {}

    def setup_pipeline(self, **kwargs) -> bool:
        """
        Set up disk-based inference pipeline system for the plugin.

        Args:
            **kwargs: Pipeline configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def execute_pipeline(self, data: Any, pipeline_config: Dict[str, Any] = None) -> Any:
        """
        Execute inference using the disk-based pipeline system.

        Args:
            data: Input data for inference
            pipeline_config: Configuration for the pipeline execution

        Returns:
            Inference results from the pipeline
        """
        # Default implementation falls back to regular inference
        return self.infer(data)

    def create_pipeline_stages(self, **kwargs) -> List['PipelineStage']:
        """
        Create pipeline stages for the model.

        Args:
            **kwargs: Stage configuration parameters

        Returns:
            List of PipelineStage objects
        """
        # Default implementation returns empty list
        return []

    def get_pipeline_manager(self):
        """
        Get the pipeline manager instance.

        Returns:
            Pipeline manager instance or None
        """
        try:
            from .disk_pipeline import PipelineManager
            return PipelineManager
        except ImportError:
            logger.warning("Pipeline module not available")
            return None

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline execution.

        Returns:
            Dictionary containing pipeline statistics
        """
        # Default implementation - return basic stats
        return {
            'pipeline_enabled': False,
            'num_stages': 0,
            'checkpoint_directory': None,
            'pipeline_performance': {}
        }

    def setup_activation_offloading(self, **kwargs) -> bool:
        """
        Set up activation offloading system for managing intermediate activations between RAM and disk.

        Args:
            **kwargs: Activation offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_activation_offloading(self, **kwargs) -> bool:
        """
        Enable activation offloading for the model to move intermediate activations between RAM and disk.

        Args:
            **kwargs: Activation offloading configuration parameters

        Returns:
            True if activation offloading was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def offload_activations(self, **kwargs) -> bool:
        """
        Offload specific activations to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def predict_activation_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which activations will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping activation names to access probabilities
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def get_activation_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about activation offloading operations.

        Returns:
            Dictionary containing activation offloading statistics
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def initialize_security(self, security_level: SecurityLevel = SecurityLevel.MEDIUM_TRUST,
                           resource_limits: Optional[ResourceLimits] = None) -> bool:
        """
        Initialize security and resource isolation for the plugin.

        Args:
            security_level: Security level for the plugin
            resource_limits: Resource limits to enforce

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Set security parameters
            self._security_level = security_level
            self._resource_limits = resource_limits

            # Initialize resource isolation
            success = initialize_plugin_isolation(
                plugin_id=self.metadata.name,
                security_level=security_level,
                resource_limits=resource_limits
            )

            if success:
                self._security_initialized = True
                logger.info(f"Security initialized for plugin {self.metadata.name} with level {security_level.value}")
            else:
                logger.error(f"Failed to initialize security for plugin {self.metadata.name}")

            return success
        except Exception as e:
            logger.error(f"Error initializing security for plugin {self.metadata.name}: {e}")
            return False

    def validate_file_access(self, file_path: str) -> bool:
        """
        Validate if the plugin is allowed to access a specific file path.

        Args:
            file_path: Path to the file to access

        Returns:
            True if access is allowed, False otherwise
        """
        if not self._security_initialized:
            logger.warning(f"Security not initialized for plugin {self.metadata.name}, allowing access by default")
            return True

        from .security_manager import validate_path_access
        return validate_path_access(self.metadata.name, file_path)

    def validate_network_access(self, host: str) -> bool:
        """
        Validate if the plugin is allowed to connect to a specific network host.

        Args:
            host: Host to connect to

        Returns:
            True if access is allowed, False otherwise
        """
        if not self._security_initialized:
            logger.warning(f"Security not initialized for plugin {self.metadata.name}, allowing access by default")
            return True

        from .security_manager import validate_network_access
        return validate_network_access(self.metadata.name, host)

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage information for the plugin.

        Returns:
            Dictionary with resource usage information
        """
        if not self._security_initialized:
            logger.warning(f"Security not initialized for plugin {self.metadata.name}")
            return {}

        from .security_manager import get_plugin_resource_usage
        return get_plugin_resource_usage(self.metadata.name)

    def cleanup_security(self) -> bool:
        """
        Clean up security and resource isolation for the plugin.

        Returns:
            True if cleanup was successful, False otherwise
        """
        if not self._security_initialized:
            logger.info(f"Security not initialized for plugin {self.metadata.name}, nothing to clean up")
            return True

        success = cleanup_plugin_isolation(self.metadata.name)

        if success:
            self._security_initialized = False
            logger.info(f"Security cleaned up for plugin {self.metadata.name}")
        else:
            logger.error(f"Failed to clean up security for plugin {self.metadata.name}")

        return success


class TextModelPluginInterface(ModelPluginInterface):
    """
    Interface for text-based model plugins in the Inference-PIO system.
    """

    def __init__(self, metadata: ModelPluginMetadata):
        super().__init__(metadata)
        if metadata.plugin_type != PluginType.MODEL_COMPONENT:
            raise ValueError(
                f"Plugin type must be MODEL_COMPONENT, got {metadata.plugin_type}"
            )

    # Methods inherited from ModelPluginInterface (via StandardPluginInterface):
    # - initialize (abstract)
    # - load_model (abstract)
    # - infer (abstract)
    # - cleanup (abstract)
    # - supports_config (abstract)

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> Any:
        """
        Tokenize the given text.

        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters

        Returns:
            Tokenized result
        """

    @abstractmethod
    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional decoding parameters

        Returns:
            Decoded text
        """

    @abstractmethod
    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt: Text generation prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """

    def chat_completion(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024, **kwargs) -> str:
        """
        Perform chat completion with the model.

        Args:
            messages: List of message dictionaries with role and content
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        # Default implementation that formats messages and calls generate_text
        formatted_prompt = self._format_chat_messages(messages)
        return self.generate_text(formatted_prompt, max_new_tokens=max_new_tokens, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for the model.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Formatted prompt string
        """
        # Default implementation - subclasses should override for model-specific formatting
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        return formatted

    def setup_adaptive_batching(self, **kwargs) -> bool:
        """
        Set up adaptive batching system for dynamic batch size adjustment.

        Args:
            **kwargs: Adaptive batching configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def get_optimal_batch_size(self, processing_time_ms: float, tokens_processed: int) -> int:
        """
        Get the optimal batch size based on current performance metrics.

        Args:
            processing_time_ms: Processing time for the current batch in milliseconds
            tokens_processed: Number of tokens processed in the current batch

        Returns:
            Recommended batch size for the next batch
        """
        # Default implementation - return a fixed batch size
        return 1

    def adjust_batch_size(self) -> Tuple[int, bool, Optional[str]]:
        """
        Adjust the batch size based on current metrics.

        Returns:
            Tuple of (new_batch_size, was_adjusted, reason_for_adjustment)
        """
        # Default implementation - return current batch size without adjustment
        return 1, False, None

    def get_batching_status(self) -> Dict[str, Any]:
        """
        Get the current status of the adaptive batching system.

        Returns:
            Dictionary containing batching status information
        """
        # Default implementation - return basic status
        return {
            'current_batch_size': 1,
            'adaptive_batching_enabled': False,
            'memory_pressure_ratio': 0.0,
            'performance_score': 0.0
        }

    def setup_distributed_simulation(self, **kwargs) -> bool:
        """
        Set up distributed simulation system for multi-GPU execution simulation.

        Args:
            **kwargs: Distributed simulation configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_distributed_execution(self, **kwargs) -> bool:
        """
        Enable distributed execution simulation on single or multiple GPUs.

        Args:
            **kwargs: Distributed execution configuration parameters

        Returns:
            True if distributed execution was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def partition_model_for_distributed(self, num_partitions: int = 1, **kwargs) -> bool:
        """
        Partition the model for distributed execution.

        Args:
            num_partitions: Number of partitions to create
            **kwargs: Additional partitioning parameters

        Returns:
            True if partitioning was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def get_distributed_simulation_manager(self):
        """
        Get the distributed simulation manager instance.

        Returns:
            Distributed simulation manager instance or None
        """
        try:
            from .distributed_simulation import DistributedSimulationManager
            return DistributedSimulationManager
        except ImportError:
            logger.warning("Distributed simulation module not available")
            return None

    def get_virtual_gpu_simulator(self):
        """
        Get the virtual GPU simulator instance.

        Returns:
            Virtual GPU simulator instance or None
        """
        try:
            from .virtual_gpu_simulation import VirtualGPUSimulator
            return VirtualGPUSimulator
        except ImportError:
            logger.warning("Virtual GPU simulation module not available")
            return None

    def execute_with_distributed_simulation(self, data: Any) -> Any:
        """
        Execute inference using distributed simulation.

        Args:
            data: Input data for inference

        Returns:
            Inference results
        """
        # Default implementation falls back to regular inference
        return self.infer(data)

    def get_distributed_stats(self) -> Dict[str, Any]:
        """
        Get statistics about distributed execution.

        Returns:
            Dictionary containing distributed execution statistics
        """
        # Default implementation - return basic stats
        return {
            'distributed_simulation_enabled': False,
            'num_partitions': 0,
            'num_virtual_gpus': 0,
            'partition_strategy': 'none',
            'memory_per_partition_gb': 0.0
        }

    def synchronize_partitions(self) -> bool:
        """
        Synchronize all model partitions.

        Returns:
            True if synchronization was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def pipeline_synchronize(self, current_stage: int, num_stages: int) -> bool:
        """
        Synchronize partitions in a pipeline fashion.

        Args:
            current_stage: Current pipeline stage
            num_stages: Total number of pipeline stages

        Returns:
            True if synchronization was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def get_synchronization_manager(self):
        """
        Get the synchronization manager instance.

        Returns:
            Synchronization manager instance or None
        """
        try:
            from .virtual_gpu_simulation import DistributedExecutionSimulator
            return DistributedExecutionSimulator
        except ImportError:
            logger.warning("Synchronization module not available")
            return None

    def setup_tensor_compression(self, **kwargs) -> bool:
        """
        Set up tensor compression system for model weights and activations.

        Args:
            **kwargs: Tensor compression configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def compress_model_weights(self, compression_ratio: float = 0.5, **kwargs) -> bool:
        """
        Compress model weights using tensor compression techniques.

        Args:
            compression_ratio: Target compression ratio (0.0 to 1.0)
            **kwargs: Additional compression parameters

        Returns:
            True if compression was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def decompress_model_weights(self) -> bool:
        """
        Decompress model weights back to original form.

        Returns:
            True if decompression was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def compress_activations(self, **kwargs) -> bool:
        """
        Compress model activations during inference.

        Args:
            **kwargs: Activation compression parameters

        Returns:
            True if activation compression was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get tensor compression statistics.

        Returns:
            Dictionary containing compression statistics
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def enable_adaptive_compression(self, **kwargs) -> bool:
        """
        Enable adaptive compression that adjusts based on available memory.

        Args:
            **kwargs: Adaptive compression configuration parameters

        Returns:
            True if adaptive compression was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def setup_disk_offloading(self, **kwargs) -> bool:
        """
        Set up disk offloading system for managing model components between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_disk_offloading(self, **kwargs) -> bool:
        """
        Enable disk offloading for the model to move parts between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if disk offloading was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def offload_model_parts(self, **kwargs) -> bool:
        """
        Offload specific model parts to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def predict_model_part_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which model parts will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping model part names to access probabilities
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def get_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about disk offloading operations.

        Returns:
            Dictionary containing offloading statistics
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def setup_activation_offloading(self, **kwargs) -> bool:
        """
        Set up activation offloading system for managing intermediate activations between RAM and disk.

        Args:
            **kwargs: Activation offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def enable_activation_offloading(self, **kwargs) -> bool:
        """
        Enable activation offloading for the model to move intermediate activations between RAM and disk.

        Args:
            **kwargs: Activation offloading configuration parameters

        Returns:
            True if activation offloading was enabled successfully, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def offload_activations(self, **kwargs) -> bool:
        """
        Offload specific activations to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return True

    def predict_activation_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which activations will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping activation names to access probabilities
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def get_activation_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about activation offloading operations.

        Returns:
            Dictionary containing activation offloading statistics
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def initialize_security(self, security_level: SecurityLevel = SecurityLevel.MEDIUM_TRUST,
                           resource_limits: Optional[ResourceLimits] = None) -> bool:
        """
        Initialize security and resource isolation for the plugin.

        Args:
            security_level: Security level for the plugin
            resource_limits: Resource limits to enforce

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Set security parameters
            self._security_level = security_level
            self._resource_limits = resource_limits

            # Initialize resource isolation
            success = initialize_plugin_isolation(
                plugin_id=self.metadata.name,
                security_level=security_level,
                resource_limits=resource_limits
            )

            if success:
                self._security_initialized = True
                logger.info(f"Security initialized for plugin {self.metadata.name} with level {security_level.value}")
            else:
                logger.error(f"Failed to initialize security for plugin {self.metadata.name}")

            return success
        except Exception as e:
            logger.error(f"Error initializing security for plugin {self.metadata.name}: {e}")
            return False

    def validate_file_access(self, file_path: str) -> bool:
        """
        Validate if the plugin is allowed to access a specific file path.

        Args:
            file_path: Path to the file to access

        Returns:
            True if access is allowed, False otherwise
        """
        if not self._security_initialized:
            logger.warning(f"Security not initialized for plugin {self.metadata.name}, allowing access by default")
            return True

        from .security_manager import validate_path_access
        return validate_path_access(self.metadata.name, file_path)

    def validate_network_access(self, host: str) -> bool:
        """
        Validate if the plugin is allowed to connect to a specific network host.

        Args:
            host: Host to connect to

        Returns:
            True if access is allowed, False otherwise
        """
        if not self._security_initialized:
            logger.warning(f"Security not initialized for plugin {self.metadata.name}, allowing access by default")
            return True

        from .security_manager import validate_network_access
        return validate_network_access(self.metadata.name, host)

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage information for the plugin.

        Returns:
            Dictionary with resource usage information
        """
        if not self._security_initialized:
            logger.warning(f"Security not initialized for plugin {self.metadata.name}")
            return {}

        from .security_manager import get_plugin_resource_usage
        return get_plugin_resource_usage(self.metadata.name)

    def cleanup_security(self) -> bool:
        """
        Clean up security and resource isolation for the plugin.

        Returns:
            True if cleanup was successful, False otherwise
        """
        if not self._security_initialized:
            logger.info(f"Security not initialized for plugin {self.metadata.name}, nothing to clean up")
            return True

        success = cleanup_plugin_isolation(self.metadata.name)

        if success:
            self._security_initialized = False
            logger.info(f"Security cleaned up for plugin {self.metadata.name}")
        else:
            logger.error(f"Failed to clean up security for plugin {self.metadata.name}")

        return success


class BaseAttention(nn.Module):
    """
    Base class for attention mechanisms in the Inference-PIO system.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for the attention mechanism.

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """


logger = logging.getLogger(__name__)


__all__ = [
    "PluginType",
    "ModelPluginMetadata",
    "ModelPluginInterface",
    "TextModelPluginInterface",
    "BaseAttention",
    "ActivationOffloadingManager",
    "ActivationPriority",
    "ActivationAccessPattern",
    "SecurityLevel",
    "ResourceLimits",
    "logger"
]