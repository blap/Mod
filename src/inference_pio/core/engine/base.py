"""
Base Model Class for Numpy Inference Engine
"""
from .layers import Module

class BaseModel(Module):
    """
    Abstract base class for all Numpy-based models in Inference-PIO.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement generate method.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method.")
