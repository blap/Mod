"""
Plugin Manager System for Inference-PIO
Dependency-Free
"""

import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..common.interfaces.improved_base_plugin_interface import ModelPluginInterface
from ..common.security.security_manager import ResourceLimits, SecurityLevel
from ..core.engine.backend import HAS_CUDA
from .base.gpu_interface import GPUHardwareInterface
from ..common.hardware.hardware_analyzer import HardwareAnalyzer

logger = logging.getLogger(__name__)

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, ModelPluginInterface] = {}
        self.active_plugins: Dict[str, ModelPluginInterface] = {}
        self.plugin_paths: List[Path] = []
        self.security_enabled = True

        # New Multi-Backend Registry
        # Stores multiple active backend instances simultaneously
        # Key: device_id (e.g., 'cuda:0', 'opencl:intel:0', 'cpu')
        self.active_backends: Dict[str, Union[GPUHardwareInterface, Any]] = {}

    def load_hardware_backends(self) -> int:
        """
        Detect hardware and load ALL appropriate self-contained plugins.
        This enables heterogeneous compute (NVIDIA + Intel + AMD + CPU).
        """
        analyzer = HardwareAnalyzer()
        info = analyzer.get_hardware_info()
        gpu_list = info.get("gpu", [])

        loaded_count = 0

        # 1. Load CPU Backend (Always available)
        self._load_cpu_backend(info.get("cpu", {}))

        # 2. Load GPU Backends
        for i, gpu_info in enumerate(gpu_list):
            vendor = gpu_info.get("vendor", "unknown").lower()
            model = gpu_info.get("name", "").lower()

            logger.info(f"Detecting hardware backend for GPU {i}: {vendor} {model}")

            backend_instance = None
            device_key = f"gpu:{i}" # Temporary key, refined by backend type later

            try:
                if "nvidia" in vendor:
                    if "10" in model or "gtx" in model:
                        from .cuda.sm61.plugin import CUDASM61Plugin
                        backend_instance = CUDASM61Plugin()
                    else:
                        from .cuda.base import CUDABasePlugin
                        backend_instance = CUDABasePlugin()
                    device_key = f"cuda:{i}"

                elif "amd" in vendor or "advanced micro devices" in vendor:
                    from .amd.base import AMDBasePlugin
                    backend_instance = AMDBasePlugin()
                    device_key = f"opencl:amd:{i}"

                elif "intel" in vendor:
                    from .intel.base import IntelBasePlugin
                    backend_instance = IntelBasePlugin()
                    device_key = f"opencl:intel:{i}"

                if backend_instance and backend_instance.initialize():
                    self.active_backends[device_key] = backend_instance
                    logger.info(f"Loaded backend {device_key}: {backend_instance.get_device_info()}")
                    loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load backend for {vendor} GPU {i}: {e}")

        return loaded_count

    def _load_cpu_backend(self, cpu_info):
        vendor = cpu_info.get("vendor_id", "").lower()
        model = cpu_info.get("brand_raw", "").lower()

        import platform
        if not vendor:
            p = platform.processor().lower()
            if "intel" in p: vendor = "intel"
            elif "amd" in p: vendor = "amd"

        backend = None
        try:
            if "intel" in vendor:
                from .cpu.intel.base import IntelCPUBasePlugin
                backend = IntelCPUBasePlugin()
            elif "amd" in vendor:
                from .cpu.amd.base import AMDCPUBasePlugin
                backend = AMDCPUBasePlugin()
            else:
                # Generic fallback?
                pass

            if backend and backend.initialize():
                self.active_backends["cpu"] = backend
                logger.info("Loaded CPU Backend")
        except Exception as e:
            logger.error(f"Failed to load CPU backend: {e}")

    # ... (Rest of the class methods remain largely unchanged, preserving existing logic) ...
    def register_plugin(self, plugin: ModelPluginInterface, name: Optional[str] = None) -> bool:
        try:
            plugin_name = name or plugin.metadata.name
            self.plugins[plugin_name] = plugin
            logger.info(f"Registered plugin: {plugin_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False

    def load_plugin_from_path(self, plugin_path: Union[str, Path]) -> bool:
        try:
            plugin_path = Path(plugin_path)
            if not plugin_path.exists() or plugin_path.suffix != ".py": return False

            plugin_dir = str(plugin_path.parent)
            if plugin_dir not in sys.path: sys.path.insert(0, plugin_dir)

            module_name = plugin_path.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, ModelPluginInterface) and obj != ModelPluginInterface and obj.__module__ == module.__name__:
                    try:
                        if hasattr(module, f"create_{obj.__name__.lower()}"):
                            factory = getattr(module, f"create_{obj.__name__.lower()}")
                            plugin = factory()
                        else:
                            plugin = obj()
                        self.register_plugin(plugin)
                    except Exception: continue
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False

    def load_plugins_from_directory(self, directory: Union[str, Path]) -> int:
        directory = Path(directory)
        if not directory.exists(): return 0
        count = 0
        for f in directory.glob("*.py"):
            if self.load_plugin_from_path(f): count += 1
        return count

    def get_plugin(self, name: str) -> Optional[ModelPluginInterface]:
        return self.plugins.get(name)

    def activate_plugin(self, name: str, **kwargs) -> bool:
        if name not in self.plugins: return False
        if name in self.active_plugins: return True

        plugin = self.plugins[name]
        try:
            if self.security_enabled:
                sec_level = kwargs.get("security_level", SecurityLevel.MEDIUM_TRUST)
                gpu_mem = 4.0 if HAS_CUDA else 0.0
                limits = kwargs.get("resource_limits", ResourceLimits(gpu_memory_gb=gpu_mem))
                if hasattr(plugin, "initialize_security"):
                    plugin.initialize_security(security_level=sec_level, resource_limits=limits)

            if not plugin._initialized:
                if not plugin.initialize(**kwargs): return False

            self.active_plugins[name] = plugin
            return True
        except Exception as e:
            logger.error(f"Activation failed: {e}")
            return False

    def deactivate_plugin(self, name: str) -> bool:
        if name not in self.active_plugins: return True
        try:
            plugin = self.active_plugins[name]
            if hasattr(plugin, "cleanup"): plugin.cleanup()
            del self.active_plugins[name]
            return True
        except Exception: return False

    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        if name not in self.active_plugins: return None
        plugin = self.active_plugins[name]
        try:
            if hasattr(plugin, "infer"): return plugin.infer(*args, **kwargs)
            if hasattr(plugin, "generate_text"): return plugin.generate_text(*args, **kwargs)
            if hasattr(plugin, "load_model"): return plugin.load_model(*args, **kwargs)
            return None
        except Exception: return None

    def discover_and_load_plugins(self, models_directory=None):
        if not models_directory:
            models_directory = Path(__file__).parent.parent / "models"
        models_directory = Path(models_directory)
        if not models_directory.exists(): return 0

        count = 0
        for model_dir in models_directory.iterdir():
            if model_dir.is_dir():
                if model_dir.name in ["language", "vision_language", "coding", "specialized"]:
                    for sub in model_dir.iterdir():
                        if sub.is_dir(): count += self._load_model_from_directory(sub)
                else:
                    count += self._load_model_from_directory(model_dir)
        return count

    def _load_model_from_directory(self, model_dir):
        plugin_file = model_dir / "plugin.py"
        if plugin_file.exists():
            if self.load_plugin_from_path(plugin_file): return 1
        return 0

_plugin_manager = None
def get_plugin_manager():
    global _plugin_manager
    if not _plugin_manager: _plugin_manager = PluginManager()
    return _plugin_manager

# Facade
def register_plugin(p, n=None): return get_plugin_manager().register_plugin(p, n)
def load_plugin_from_path(p): return get_plugin_manager().load_plugin_from_path(p)
def load_plugins_from_directory(d): return get_plugin_manager().load_plugins_from_directory(d)
def activate_plugin(n, **k): return get_plugin_manager().activate_plugin(n, **k)
def execute_plugin(n, *a, **k): return get_plugin_manager().execute_plugin(n, *a, **k)
def discover_and_load_plugins(d=None): return get_plugin_manager().discover_and_load_plugins(d)

__all__ = ["PluginManager", "get_plugin_manager", "register_plugin", "load_plugin_from_path",
           "load_plugins_from_directory", "activate_plugin", "execute_plugin", "discover_and_load_plugins"]
