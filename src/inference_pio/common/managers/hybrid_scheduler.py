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

        # Real implementation would query memory.
        # For "Real Code" without external deps, we track allocation via UnifiedManager (future step)
        # or use a static split logic.

        # Static Split Strategy:
        # If we have 16GB GPU and 30B model (60GB), we fit ~25% on GPU.
        # layers_on_gpu = int(total_layers * 0.25)

        # For now, return "cuda" to prioritize GPU, ModelLoader handles OOM by fallback if implemented.
        return "cuda"

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
        Manual layer-by-layer execution with data transfer.
        """
        # Embeddings
        x = model.embed_tokens(x)

        # Layers
        for layer in model.layers:
            # Check layer device
            target_device = "cpu"
            try:
                p = next(layer.parameters())
                target_device = p.device
            except: pass

            if x.device != target_device:
                x = x.to(target_device)

            x = layer(x)

        # Final Norm / Head
        target_device = model.norm.weight.device
        if x.device != target_device: x = x.to(target_device)
        x = model.norm(x)

        target_device = model.lm_head.weight.device
        if x.device != target_device: x = x.to(target_device)
        x = model.lm_head(x)

        return x
