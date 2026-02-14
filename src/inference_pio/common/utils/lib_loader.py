import os
import sys
import ctypes
import logging

logger = logging.getLogger(__name__)

def load_backend_lib(backend_name: str, specific_target: str = None) -> ctypes.CDLL:
    """
    Standardized loader for backend shared libraries.
    Resolves paths for Windows (.dll) and Linux (.so).
    """
    if specific_target:
        lib_name = f"libtensor_ops_{specific_target}"
    else:
        lib_name = "libtensor_ops"

    if sys.platform == "win32":
        lib_file = f"{lib_name}.dll"
    else:
        lib_file = f"{lib_name}.so"

    # Path Resolution Logic
    # 1. Check relative to this file (development/source)
    # ../../plugins/{backend_name}/c_src/
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "plugins"))

    # Mapping for backend directory names
    backend_dir_map = {
        "cpu": "cpu",
        "cuda": "cuda",
        "amd": "amd",
        "intel": "intel"
    }

    target_dir = backend_dir_map.get(backend_name, backend_name)
    path = os.path.join(base_path, target_dir, "c_src", lib_file)

    if not os.path.exists(path):
        # Fallback 2: Check standard install location /usr/local/lib/inference_pio/
        path = os.path.join("/usr/local/lib/inference_pio", lib_file)

    if not os.path.exists(path):
        # Fallback 3: Current directory
        path = lib_file

    try:
        return ctypes.CDLL(path)
    except OSError as e:
        msg = f"Failed to load backend library for {backend_name} ({specific_target}) at {path}: {e}"
        raise ImportError(msg) from e
