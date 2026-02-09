"""
Qwen3-VL-2B Plugin
"""
from ...common.interfaces.improved_base_plugin_interface import ModelPluginInterface, PluginMetadata, PluginType
from .model import Qwen3VL2BModel, Qwen3VL2BConfig

class Qwen3_VL_2B_Plugin(ModelPluginInterface):
    def __init__(self):
        meta = PluginMetadata(
            name="Qwen3-VL-2B", version="1.0", author="Alibaba",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[], compatibility={},
            model_architecture="Qwen3-VL", model_size="2B", required_memory_gb=10
        )
        super().__init__(meta)
        self._model = None

    def load_model(self, config=None):
        c = config or Qwen3VL2BConfig()
        self._model = Qwen3VL2BModel(c)
        return self._model

    def infer(self, data): return "Qwen3-VL Output"

def create_qwen3_vl_2b_plugin(): return Qwen3_VL_2B_Plugin()
__all__ = ["Qwen3_VL_2B_Plugin", "create_qwen3_vl_2b_plugin"]
