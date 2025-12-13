"""
Model loading and initialization system that handles different formats and sizes.

This module provides model loading and initialization that handles different formats and sizes.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union, Type
from pathlib import Path
import logging
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model


class ModelLoader:
    """
    System for loading and initializing models with support for different formats and sizes.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def load_model(
        self,
        model_name_or_path: str,
        model_class: Type[nn.Module],
        config_class: Type[Any],
        config: Optional[Dict[str, Any]] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Load a model from a path or Hugging Face model hub.
        
        Args:
            model_name_or_path: Path to model or Hugging Face model name
            model_class: Model class to instantiate
            config_class: Configuration class to use
            config: Configuration dictionary
            device_map: Device mapping for model sharding
            torch_dtype: Data type for model weights
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        config = config or {}
        
        # Convert torch_dtype string to actual dtype if needed
        if isinstance(torch_dtype, str):
            torch_dtype = self._str_to_dtype(torch_dtype)
        
        # Load configuration
        model_config = self._load_config(model_name_or_path, config_class, config)
        
        # Determine if we're loading from local path or Hugging Face hub
        if self._is_local_path(model_name_or_path):
            return self._load_from_local_path(
                model_name_or_path, model_class, model_config, 
                device_map, torch_dtype, **kwargs
            )
        else:
            return self._load_from_hub(
                model_name_or_path, model_class, model_config,
                device_map, torch_dtype, **kwargs
            )
    
    def _load_config(
        self, 
        model_name_or_path: str, 
        config_class: Type[Any], 
        config: Dict[str, Any]
    ) -> Any:
        """Load model configuration."""
        # Try to load from local path first
        if self._is_local_path(model_name_or_path):
            config_path = Path(model_name_or_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    local_config = json.load(f)
                local_config.update(config)
                return config_class(**local_config)
        
        # For Hugging Face models, try to load config from hub
        try:
            config_obj = config_class.from_pretrained(model_name_or_path, **config)
            return config_obj
        except Exception:
            # If that fails, create with provided config
            return config_class(**config)
    
    def _is_local_path(self, path: str) -> bool:
        """Check if path is a local path."""
        path_obj = Path(path)
        return path_obj.exists() or os.path.isdir(path) or os.path.isfile(path)
    
    def _load_from_local_path(
        self,
        model_path: str,
        model_class: Type[nn.Module],
        config: Any,
        device_map: Optional[Union[str, Dict[str, Any]]],
        torch_dtype: Optional[torch.dtype],
        **kwargs
    ) -> nn.Module:
        """Load model from local path."""
        model_path = Path(model_path)
        
        # Check for different model formats
        if (model_path / "model.safetensors").exists():
            return self._load_safetensors_model(
                model_path, model_class, config, device_map, torch_dtype, **kwargs
            )
        elif (model_path / "pytorch_model.bin").exists():
            return self._load_torch_model(
                model_path, model_class, config, device_map, torch_dtype, **kwargs
            )
        elif (model_path / "model.onnx").exists():
            return self._load_onnx_model(
                model_path, model_class, config, device_map, torch_dtype, **kwargs
            )
        else:
            # Try to load with transformers
            try:
                return model_class.from_pretrained(
                    model_path,
                    config=config,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            except Exception as e:
                self._logger.error(f"Failed to load model from {model_path}: {e}")
                raise
    
    def _load_from_hub(
        self,
        model_name: str,
        model_class: Type[nn.Module],
        config: Any,
        device_map: Optional[Union[str, Dict[str, Any]]],
        torch_dtype: Optional[torch.dtype],
        **kwargs
    ) -> nn.Module:
        """Load model from Hugging Face hub."""
        try:
            return model_class.from_pretrained(
                model_name,
                config=config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                **kwargs
            )
        except Exception as e:
            self._logger.error(f"Failed to load model {model_name} from hub: {e}")
            
            # Try downloading first and then loading locally
            try:
                local_path = snapshot_download(model_name)
                return self._load_from_local_path(
                    local_path, model_class, config, device_map, torch_dtype, **kwargs
                )
            except Exception as e2:
                self._logger.error(f"Failed to download and load model {model_name}: {e2}")
                raise
    
    def _load_safetensors_model(
        self,
        model_path: Path,
        model_class: Type[nn.Module],
        config: Any,
        device_map: Optional[Union[str, Dict[str, Any]]],
        torch_dtype: Optional[torch.dtype],
        **kwargs
    ) -> nn.Module:
        """Load model from safetensors format."""
        model = model_class(config)
        model_file = model_path / "model.safetensors"
        
        # Load the model weights
        load_safetensors_model(model, str(model_file))
        
        # Move to appropriate device
        if device_map:
            if isinstance(device_map, str) and device_map in ['auto', 'balanced']:
                model = self._auto_device_map(model, device_map)
            else:
                model = model.to(device_map if isinstance(device_map, str) else next(iter(device_map.values())))
        
        # Convert dtype if needed
        if torch_dtype:
            model = model.to(torch_dtype)
        
        return model
    
    def _load_torch_model(
        self,
        model_path: Path,
        model_class: Type[nn.Module],
        config: Any,
        device_map: Optional[Union[str, Dict[str, Any]]],
        torch_dtype: Optional[torch.dtype],
        **kwargs
    ) -> nn.Module:
        """Load model from torch format."""
        model = model_class(config)
        model_file = model_path / "pytorch_model.bin"
        
        # Load the model weights
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Move to appropriate device
        if device_map:
            if isinstance(device_map, str) and device_map in ['auto', 'balanced']:
                model = self._auto_device_map(model, device_map)
            else:
                model = model.to(device_map if isinstance(device_map, str) else next(iter(device_map.values())))
        
        # Convert dtype if needed
        if torch_dtype:
            model = model.to(torch_dtype)
        
        return model
    
    def _load_onnx_model(
        self,
        model_path: Path,
        model_class: Type[nn.Module],
        config: Any,
        device_map: Optional[Union[str, Dict[str, Any]]],
        torch_dtype: Optional[torch.dtype],
        **kwargs
    ) -> nn.Module:
        """Load model from ONNX format."""
        import onnxruntime as ort

        model_file = model_path / "model.onnx"

        # Check if the model file exists
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_file}")

        # Create ONNX Runtime session
        session_options = ort.SessionOptions()

        # Set optimization level
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Determine execution provider based on device_map
        if device_map and 'cuda' in str(device_map).lower():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Create the ONNX session
        try:
            session = ort.InferenceSession(
                str(model_file),
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX Runtime session: {e}")

        # Create a wrapper class to make ONNX model compatible with PyTorch interface
        class ONNXModelWrapper(nn.Module):
            def __init__(self, session):
                super().__init__()
                self.session = session
                self.config = config  # Store config for compatibility

            def forward(self, *args, **kwargs):
                # Convert PyTorch tensors to numpy arrays
                # Handle both positional and keyword arguments
                inputs = {}

                # Add positional arguments to inputs if they exist
                input_names = [input.name for input in self.session.get_inputs()]

                # If we have positional args and input names match a standard pattern
                if args and len(args) <= len(input_names):
                    for i, arg in enumerate(args):
                        if i < len(input_names):
                            inputs[input_names[i]] = arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg

                # Add keyword arguments to inputs
                for k, v in kwargs.items():
                    if v is not None:
                        inputs[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v

                # Run inference
                try:
                    outputs = self.session.run(None, inputs)
                except Exception as e:
                    raise RuntimeError(f"ONNX inference failed: {e}")

                # Convert back to PyTorch tensors
                result = tuple(torch.from_numpy(output) for output in outputs)

                # Return as BaseModelOutput-like object or just the tensor
                if len(result) == 1:
                    return result[0]
                else:
                    return result

            def generate(self, *args, **kwargs):
                # For generation, we'll call forward with the appropriate inputs
                return self.forward(*args, **kwargs)

        # Create and return the wrapper
        onnx_model = ONNXModelWrapper(session)

        return onnx_model
    
    def _auto_device_map(self, model: nn.Module, strategy: str = 'auto') -> nn.Module:
        """Apply automatic device mapping to model."""
        if torch.cuda.device_count() > 1:
            # Use model parallelism for multi-GPU setups
            from transformers import dispatch_model
            # This is a simplified approach - actual implementation would be more complex
            pass
        
        # For now, just return the model as is
        return model
    
    def _str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string to torch dtype."""
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float64': torch.float64,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
        }
        
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        
        return dtype_map[dtype_str]
    
    def save_model(
        self,
        model: nn.Module,
        save_path: Union[str, Path],
        format: str = 'torch',  # 'torch', 'safetensors', 'onnx'
        **kwargs
    ) -> None:
        """
        Save model in specified format.
        
        Args:
            model: Model to save
            save_path: Path to save the model
            format: Format to save in ('torch', 'safetensors', 'onnx')
            **kwargs: Additional arguments
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'safetensors':
            model_file = save_path / "model.safetensors"
            save_safetensors_model(model, str(model_file))
        elif format == 'torch':
            model_file = save_path / "pytorch_model.bin"
            torch.save(model.state_dict(), model_file)
        elif format == 'onnx':
            import torch.onnx
            model_file = save_path / "model.onnx"

            # Prepare dummy inputs for ONNX export - try to infer from model if possible
            # First, try to use provided sample inputs from kwargs
            sample_inputs = kwargs.get('sample_inputs', None)

            if sample_inputs is not None:
                # Use provided sample inputs
                if isinstance(sample_inputs, (list, tuple)):
                    args = sample_inputs
                else:
                    # If it's a single tensor or dict, handle accordingly
                    args = (sample_inputs,) if not isinstance(sample_inputs, dict) else (sample_inputs,)
            else:
                # Try to infer input signature from model's forward method or use default
                # For general models, we'll try a simple tensor input
                try:
                    # Try to run the model with a simple tensor first to see if it works
                    test_input = torch.randn(1, 10)  # Default size for testing
                    _ = model(test_input)  # Test run to verify model works
                    args = (test_input,)
                    input_names = ['input']
                except:
                    # If simple tensor doesn't work, try to use more complex default inputs
                    # based on the Qwen model signature
                    dummy_input_ids = torch.randint(0, 1000, (1, 10))
                    dummy_pixel_values = torch.randn(1, 3, 224, 224)
                    dummy_attention_mask = torch.ones(1, 10)
                    args = (dummy_input_ids, dummy_pixel_values, dummy_attention_mask)
                    input_names = ['input_ids', 'pixel_values', 'attention_mask']

            # Determine input names and dynamic axes based on the arguments
            if len(args) == 1:
                input_names = ['input']
                dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            elif len(args) == 3:
                input_names = ['input_ids', 'pixel_values', 'attention_mask']
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'pixel_values': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'output': {0: 'batch_size'}
                }
            else:
                # For other cases, use generic names
                input_names = [f'input_{i}' for i in range(len(args))]
                dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
                dynamic_axes['output'] = {0: 'batch_size'}

            # Export the model to ONNX format
            try:
                torch.onnx.export(
                    model,
                    args,
                    str(model_file),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=['output'],
                    dynamic_axes=dynamic_axes
                )
            except Exception as e:
                raise RuntimeError(f"Failed to export model to ONNX format: {e}")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self._logger.info(f"Model saved to {save_path} in {format} format")


# Global model loader instance
model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """
    Get the global model loader instance.
    
    Returns:
        ModelLoader instance
    """
    return model_loader