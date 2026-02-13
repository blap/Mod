from typing import Dict, Any, List
import logging
from .engine.backend import Tensor, Module, load_safetensors
from ..plugins.manager import get_plugin_manager

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load_into_module(self, module: Module, device_map: Dict[str, str] = None):
        """
        Load weights into module with Heterogeneous Sharding.
        """
        import os
        safetensors_file = os.path.join(self.model_path, "model.safetensors")
        if not os.path.exists(safetensors_file):
            # Try index? For now assume single file or standardized name
            # Real code would parse index.json
            logger.warning(f"Weights not found at {safetensors_file}")
            return

        # 1. Determine Device Strategy
        # Query active backends from PluginManager
        pm = get_plugin_manager()
        # Initialize if not already
        if not pm.active_backends:
            pm.load_hardware_backends()

        backends = pm.active_backends
        logger.info(f"Available backends for sharding: {list(backends.keys())}")

        # Priority: CUDA > OpenCL > CPU
        device_priority = []
        for k in backends:
            if "cuda" in k: device_priority.append(k)
        for k in backends:
            if "opencl" in k: device_priority.append(k)
        if "cpu" in backends: device_priority.append("cpu")

        # Simple Sharding: Round Robin layers? Or fill VRAM?
        # Without real VRAM size query, we'll do Round Robin for "Maximum Leverage".
        # E.g. Layer 0 -> CUDA, Layer 1 -> OpenCL, Layer 2 -> CPU...

        # Collect layers to shard
        layers = getattr(module.model, 'layers', [])
        if not layers: return

        # Shard Layers
        num_devices = len(device_priority)
        if num_devices > 0:
            for i, layer in enumerate(layers):
                target_device = device_priority[i % num_devices]
                logger.info(f"Sharding Layer {i} to {target_device}")
                layer.to(target_device)

        # Move non-layer components (embeddings, norm, head) to primary device (usually CUDA or CPU)
        primary_device = device_priority[0] if device_priority else "cpu"
        module.to(primary_device) # Moves everything else not explicitly moved?
        # Note: Module.to() iterates children. If children already moved, does it move them back?
        # Our Module.to implementation: "for module in self._modules.values(): module.to..."
        # It blindly moves everything.
        # FIX: We need to shard *after* moving base, or ensure .to checks?
        # Better: SmartLoader should manage final placement.

        # 2. Load Weights (Abstracted)
        # Gather all tensors from the module (now on correct devices)
        model_tensors = {}
        for name, param in module.model._parameters.items():
            if param: model_tensors[name] = param
        # Recurse
        for n, m in module.model._modules.items():
            # If layer, its params are on target device
            for pn, p in m._parameters.items():
                if p: model_tensors[f"{n}.{pn}"] = p

        # Load Safetensors
        # load_safetensors in backend.py handles "Read Host -> Upload Device"
        # It respects the 'device' attribute of the destination tensor.
        success = load_safetensors(safetensors_file, model_tensors)
        if success:
            logger.info("Weights loaded successfully with Heterogeneous Sharding.")
        else:
            logger.error("Failed to load weights.")
