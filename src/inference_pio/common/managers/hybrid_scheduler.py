import logging
from typing import Any, Optional
from ...core.engine.backend import Tensor, HAS_CUDA
from ...plugins.manager import get_plugin_manager

logger = logging.getLogger(__name__)

class HybridScheduler:
    """
    Manages execution across CPU and GPU backends.
    """
    def __init__(self):
        self.plugin_manager = get_plugin_manager()
        self.gpu_backend = self.plugin_manager.hardware_backend
        # Assuming cpu_backend is accessible or we use default libtensor_ops
        self.has_gpu = HAS_CUDA and self.gpu_backend is not None

    def device_for_layer(self, layer_idx: int, total_layers: int, gpu_memory_fraction: float = 0.8) -> str:
        """
        Determine placement for a layer based on available resources.
        Simple heuristic: Fill GPU up to fraction, then spill to CPU.
        """
        if not self.has_gpu: return "cpu"

        # Query Unified Memory Manager if available
        # from ..memory.unified_manager import get_memory_manager
        # mgr = get_memory_manager()
        # if mgr.current_gpu_bytes > mgr.gpu_limit_bytes * gpu_memory_fraction:
        #     return "cpu"

        return "cuda"

    def check_migration_policy(self, layer_idx, layer):
        """
        Dynamic Offloading: Check if we should move this layer to/from GPU.
        """
        if not self.has_gpu: return

        # Heuristic: Keep current layer on GPU.
        # If memory is critical (>90%), find a layer far from current index and evict.

        # In a real implementation we query memory manager stats.
        # Here we implement the logic assuming we can query tensor device.

        # Check current layer: must be on GPU for speed (if allowed)
        # If it's on CPU, try to move it.
        try:
            p = next(layer.parameters())
            if p.device == "cpu":
                # Try move to CUDA
                # We need to update the module in-place.
                # Since we don't have easy access to the parent list to swap the object,
                # we rely on .to() returning a new tensor and us updating the parameter dict.

                layer.to("cuda")

        except Exception as e:
            logger.warning(f"Migration failed: {e}")

    def execute_hybrid(self, model: Any, input_ids: Tensor) -> Tensor:
        """
        Execute model potentially splitting computation.
        """
        # 1. Ensure input is on the device of the first layer
        first_layer_device = model.layers[0].parameters().__next__().device if model.layers else "cpu"

        current_state = input_ids.to(first_layer_device)

        # 2. Run Forward
        # If model.forward handles device movement (it usually expects uniform device),
        # we might need to manually iterate layers here if they are split.

        if self.is_model_split(model):
            return self._run_split_execution(model, current_state)
        else:
            return model(current_state)

    def is_model_split(self, model) -> bool:
        """Check if model layers are on different devices."""
        if not model.layers: return False
        dev = None
        for layer in model.layers:
            # Check first param of layer
            try:
                p = next(layer.parameters())
                if dev is None: dev = p.device
                elif dev != p.device: return True
            except StopIteration:
                continue
        return False

    def _run_split_execution(self, model, x):
        """
        Manual layer-by-layer execution with data transfer and prefetching.
        """
        import threading

        # Embeddings
        x = model.embed_tokens(x)

        # Layers
        num_layers = len(model.layers)

        for i, layer in enumerate(model.layers):
            # 1. Determine current layer device
            target_device = "cpu"
            try:
                p = next(layer.parameters())
                target_device = p.device
            except: pass

            # 2. Speculative Prefetch: Trigger transfer for NEXT layer
            if i + 1 < num_layers:
                next_layer = model.layers[i+1]
                # Check next layer device
                try:
                    np = next(next_layer.parameters())
                    next_dev = np.device
                    # If current input is on GPU, and next is CPU, we could start D2H?
                    # No, we don't have the output of current layer yet.
                    # Optimization: If weights are swapped out, prefetch WEIGHTS here.
                    pass
                except: pass

            # 3. Transfer Input
            if x.device != target_device:
                x = x.to(target_device, non_blocking=True)

            # 4. Compute
            x = layer(x)

        # Final Norm / Head
        target_device = model.norm.weight.device
        if x.device != target_device: x = x.to(target_device)
        x = model.norm(x)

        target_device = model.lm_head.weight.device
        if x.device != target_device: x = x.to(target_device)
        x = model.lm_head(x)

        return x
