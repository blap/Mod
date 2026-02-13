"""
Explicit Pipeline Parallelism Utilities
Dependency-Free
"""

from typing import List, Any
import logging
from ..engine.backend import Module, Tensor

logger = logging.getLogger(__name__)

class PipelineStage:
    """
    Represents a stage in a pipeline parallel execution model.
    A stage contains a sequence of layers residing on a specific device.
    """
    def __init__(self, layers: List[Module], device: str, stage_id: int):
        self.layers = layers
        self.device = device
        self.stage_id = stage_id

        # Move layers to device immediately upon stage creation
        for layer in self.layers:
            layer.to(self.device)

    def forward(self, hidden_states: Tensor, **kwargs) -> Tensor:
        """
        Execute the stage. Automatically handles input transfer if needed.
        """
        # Ensure input is on device
        if hidden_states.device != self.device:
            # Explicit transfer (Pipeline boundary)
            hidden_states = hidden_states.to(self.device)

        for layer in self.layers:
            if hasattr(layer, 'forward'):
                # Handle varying signatures (some layers take cache, some don't)
                # For simplicity in this utility, we assume standard call
                # Real usage requires strict signature matching or kwargs unpacking
                res = layer(hidden_states, **kwargs)
                if isinstance(res, tuple):
                    hidden_states = res[0] # Assuming first return is hidden states
                else:
                    hidden_states = res

        return hidden_states

class PipelineExecutor:
    """
    Executes a sequence of PipelineStages.
    This implements sequential pipeline parallelism (sharding).
    Future versions can implement concurrent micro-batching here.
    """
    def __init__(self, stages: List[PipelineStage]):
        self.stages = sorted(stages, key=lambda s: s.stage_id)

    def forward(self, input_tensor: Tensor, **kwargs) -> Tensor:
        x = input_tensor
        for stage in self.stages:
            x = stage.forward(x, **kwargs)
        return x
