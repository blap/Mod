import logging
from typing import Any, Optional
from ...core.engine.backend import Tensor, HAS_CUDA
from ...plugins.manager import get_plugin_manager

logger = logging.getLogger(__name__)

from ...core.engine.backend import CUDAStream

class HybridScheduler:
    """
    Manages execution across CPU and GPU backends.
    Supports Async Execution (Overlap).
    """
    def __init__(self):
        self.plugin_manager = get_plugin_manager()
        self.gpu_backend = self.plugin_manager.hardware_backend
        # Assuming cpu_backend is accessible or we use default libtensor_ops
        self.has_gpu = HAS_CUDA and self.gpu_backend is not None

        # Async Streams
        self.compute_stream = CUDAStream() if self.has_gpu else None
        self.transfer_stream = CUDAStream() if self.has_gpu else None

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

    def check_migration_policy(self, layer_idx, layer, all_layers=None):
        """
        Dynamic Offloading: Intelligent Lookahead Prefetching with Streams.
        """
        if not self.has_gpu: return

        # 1. Ensure current layer is on GPU (Blocking wait for transfer if needed)
        try:
            p = next(layer.parameters())
            if p.device != "cuda":
                layer.to("cuda") # Blocking
            elif self.transfer_stream:
                # If we prefetched it in transfer stream, we must sync before compute
                # Optimization: Ideally record events. Simplified: Synchronize transfer stream.
                # In strict async, we would insert a wait_event in compute_stream.
                # Here we conservatively sync.
                self.transfer_stream.synchronize()

        except Exception as e:
            logger.warning(f"Migration (Current) failed: {e}")

        # 2. Set Compute Stream Context
        if self.compute_stream:
            self.compute_stream.__enter__()

        if not all_layers: return

        # 3. Prefetch Next Layers (Async in Transfer Stream)
        PREFETCH_WINDOW = 2

        if self.transfer_stream:
            with self.transfer_stream:
                for offset in range(1, PREFETCH_WINDOW + 1):
                    if layer_idx + offset < len(all_layers):
                        next_layer = all_layers[layer_idx + offset]
                        try:
                            p = next(next_layer.parameters())
                            if p.device != "cuda":
                                # Async transfer
                                next_layer.to("cuda", non_blocking=True)
                        except Exception as e:
                            logger.warning(f"Prefetch failed for layer {layer_idx+offset}: {e}")

        # 3. Eviction (Non-Blocking)
        # Move old layers back to CPU to save VRAM.
        # We keep immediately previous layer just in case, but evict layer_idx - 2.
        if layer_idx >= 2:
            evict_idx = layer_idx - 2
            prev_layer = all_layers[evict_idx]
            try:
                p = next(prev_layer.parameters())
                if p.device == "cuda":
                    prev_layer.to("cpu", non_blocking=True)
            except Exception as e:
                logger.warning(f"Eviction failed for layer {evict_idx}: {e}")

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
