"""
Structured Pruning - C-Engine Compatible
"""
from enum import Enum
from ...core.engine.layers import Module

class PruningMethod(Enum):
    LAYER_REMOVAL = "layer_removal"
    BLOCK_REMOVAL = "block_removal"
    HEAD_REMOVAL = "head_removal"
    MLP_REMOVAL = "mlp_removal"
    ADAPTIVE_PRUNING = "adaptive_pruning"

def apply_structured_pruning_to_model(model: Module, pruning_ratio: float, method: PruningMethod, block_size: int = 1):
    # Basic implementation: Remove layers if method is LAYER_REMOVAL
    if method == PruningMethod.LAYER_REMOVAL:
        if hasattr(model, "layers"):
            # Keep top (1-ratio) layers
            keep_count = int(len(model.layers) * (1.0 - pruning_ratio))
            # Slice ModuleList (assuming it behaves like list)
            # ModuleList in C-Engine needs slicing support or rebuild
            # Rebuild:
            from ...core.engine.layers import ModuleList
            new_layers = ModuleList(model.layers._modules_list[:keep_count])
            model.layers = new_layers

    return model
