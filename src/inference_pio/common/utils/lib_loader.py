import os
import ctypes
import sys

def get_shared_lib_extension():
    return ".dll" if sys.platform == "win32" else ".so"

def get_backend_lib_path(backend_name: str, specific_target: str = None) -> str:
    """
    Resolves the path to the backend shared library.
    Args:
        backend_name: 'cpu', 'cuda', 'amd', 'intel'
        specific_target: 'sm61', 'rx550', etc. (Optional suffix)
    """
    # Base path relative to this file (assumed to be in src/inference_pio/common/utils/lib_loader.py)
    # Actually, we need to find the root.
    # build_ops.py puts them in src/inference_pio/plugins/<backend>/c_src/

    # We assume this function is called from within the package structure.
    # Let's find the 'src' root or 'inference_pio' root.

    current_file = os.path.abspath(__file__)
    # Go up until we find 'inference_pio'
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file))) # common -> inference_pio -> src (?)
    # Actually simpler: relative to package anchor.

    # Path format: plugins/<backend>/c_src/libtensor_ops_<backend>[_<target>].<ext>
    # Exception: cpu -> libtensor_ops.so (no suffix)

    filename_base = "libtensor_ops"
    if backend_name != "cpu":
        filename_base += f"_{backend_name}"

    if specific_target:
        filename_base += f"_{specific_target}"

    ext = get_shared_lib_extension()
    filename = filename_base + ext

    # Construct absolute path
    # Assuming standard install layout
    lib_path = os.path.join(root_dir, "plugins", backend_name, "c_src", filename)

    if not os.path.exists(lib_path):
        # Fallback for CPU if specific failed? Or just return raw path for error reporting
        pass

    return lib_path

def load_backend_lib(backend_name: str, specific_target: str = None) -> ctypes.CDLL:
    path = get_backend_lib_path(backend_name, specific_target)
    try:
        lib = ctypes.CDLL(path)
        return lib
    except OSError as e:
        raise ImportError(f"Failed to load backend library for {backend_name} ({specific_target}) at {path}: {e}")
