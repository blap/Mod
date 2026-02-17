import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

# Configuration
SRC_DIR = Path("src/inference_pio")
PLUGINS_DIR = SRC_DIR / "plugins"
COMMON_C_SRC = PLUGINS_DIR / "common" / "c_src"

class Compiler:
    def compile(self, sources, output_path, include_dirs=None, defines=None, flags=None, libraries=None):
        raise NotImplementedError

class GccCompiler(Compiler):
    def __init__(self, cmd="gcc"):
        self.cmd = cmd

    def compile(self, sources, output_path, include_dirs=None, defines=None, flags=None, libraries=None):
        cmd = self.cmd.split() if isinstance(self.cmd, str) else self.cmd
        cmd += [str(s) for s in sources]

        # Base Flags
        cmd += ["-O3", "-fPIC", "-shared", "-Wall", "-Wextra", "-fopenmp"]

        if flags: cmd += flags
        if include_dirs: cmd += [f"-I{d}" for d in include_dirs]
        if defines: cmd += [f"-D{d}" for d in defines]

        cmd += ["-o", str(output_path)]

        if libraries: cmd += [f"-l{l}" for l in libraries]

        # Link math lib on Linux/standard GCC (heuristic)
        is_mingw = "mingw" in str(cmd)
        is_windows_host = platform.system() == "Windows"

        # If we are compiling FOR linux (not mingw), add -lm
        if not is_mingw:
             cmd.append("-lm")

        return run_command(cmd)

class WslCompiler(GccCompiler):
    """
    Wraps GCC invocation inside WSL (Windows Subsystem for Linux).
    Requires 'wsl' to be available in PATH.
    Path conversion (C:\\Path -> /mnt/c/Path) is handled.
    """
    def __init__(self, distro=None):
        base_cmd = ["wsl"]
        if distro:
            base_cmd += ["-d", distro]
        base_cmd += ["gcc"]
        super().__init__(base_cmd)

    def _to_wsl_path(self, path):
        # Convert Windows Path C:\Foo\Bar to /mnt/c/Foo/Bar
        # This is a naive implementation; `wsl wslpath` is safer but slower to invoke every time.
        p = Path(path).resolve()
        drive = p.drive.lower().replace(":", "")
        rel_path = p.relative_to(p.drive).as_posix()
        return f"/mnt/{drive}/{rel_path}"

    def compile(self, sources, output_path, include_dirs=None, defines=None, flags=None, libraries=None):
        # Translate paths to WSL format
        wsl_sources = [self._to_wsl_path(s) for s in sources]
        wsl_output = self._to_wsl_path(output_path)

        # Construct command directly since parent expects host paths
        cmd = self.cmd + wsl_sources

        cmd += ["-O3", "-fPIC", "-shared", "-Wall", "-Wextra", "-fopenmp"]

        if flags: cmd += flags
        if include_dirs: cmd += [f"-I{self._to_wsl_path(d)}" for d in include_dirs]
        if defines: cmd += [f"-D{d}" for d in defines]

        cmd += ["-o", wsl_output]

        if libraries: cmd += [f"-l{l}" for l in libraries]
        cmd.append("-lm") # Linux always needs math

        return run_command(cmd)


class MsvcCompiler(Compiler):
    def __init__(self, cmd="cl"):
        self.cmd = cmd

    def compile(self, sources, output_path, include_dirs=None, defines=None, flags=None, libraries=None):
        # MSVC flags
        cmd = [self.cmd] + [str(s) for s in sources]
        cmd += ["/O2", "/LD", "/openmp"] # /LD = DLL

        if flags: cmd += flags
        if include_dirs: cmd += [f"/I{d}" for d in include_dirs]
        if defines: cmd += [f"/D{d}" for d in defines]

        cmd += [f"/Fe:{output_path}"]

        if libraries: cmd += [f"{l}.lib" for l in libraries]

        return run_command(cmd)

def run_command(cmd, cwd=None):
    # Flatten if cmd contains sub-lists (e.g. from split)
    flat_cmd = []
    for item in cmd:
        if isinstance(item, list): flat_cmd.extend(item)
        else: flat_cmd.append(str(item))

    print(f"Running: {' '.join(flat_cmd)}")
    try:
        subprocess.check_call(flat_cmd, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Command not found: {flat_cmd[0]}")
        return False

def get_compilers():
    compilers = []

    # 1. Host Native
    if os.name == "nt":
        if shutil.which("cl"):
            compilers.append({"name": "Windows (MSVC)", "compiler": MsvcCompiler("cl"), "ext": ".dll", "os": "windows"})
        elif shutil.which("gcc"):
             # MinGW on Windows
             compilers.append({"name": "Windows (MinGW)", "compiler": GccCompiler("gcc"), "ext": ".dll", "os": "windows"})
    else:
        if shutil.which("gcc"):
            compilers.append({"name": "Linux (GCC)", "compiler": GccCompiler("gcc"), "ext": ".so", "os": "linux"})

    # 2. Cross Compilation
    if os.name != "nt":
        # Linux -> Windows
        cross_gcc = "x86_64-w64-mingw32-gcc"
        if shutil.which(cross_gcc):
            print(f"Found Cross-Compiler: {cross_gcc}")
            compilers.append({"name": "Windows (MinGW Cross)", "compiler": GccCompiler(cross_gcc), "ext": ".dll", "os": "windows"})
        else:
            print(f"Cross-Compiler {cross_gcc} not found. Skipping Windows build on Linux.")
    else:
        # Windows -> Linux (WSL)
        if shutil.which("wsl"):
             print(f"Found WSL. Attempting to use for Linux build.")
             try:
                 # Check if gcc exists in default distro
                 subprocess.check_call(["wsl", "gcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 compilers.append({"name": "Linux (WSL)", "compiler": WslCompiler(), "ext": ".so", "os": "linux"})
             except subprocess.CalledProcessError:
                 print("WSL found but 'gcc' not available inside it. Install gcc in WSL to build Linux binaries.")
        else:
            print("WSL not found. Skipping Linux build on Windows.")

    return compilers

def compile_cpu(compiler_info):
    name = compiler_info["name"]
    compiler = compiler_info["compiler"]
    ext = compiler_info["ext"]

    print(f"\n[Build] Compiling CPU Backend for {name}...")
    src = PLUGINS_DIR / "cpu" / "c_src" / "tensor_ops.c"
    loader_src = PLUGINS_DIR / "cpu" / "c_src" / "safetensors_loader.c"

    if not src.exists():
        print(f"Skipping: {src} not found")
        return

    out_path = src.parent / f"libtensor_ops{ext}"
    sources = [src]
    if loader_src.exists():
        sources.append(loader_src)

    if compiler.compile(sources, out_path):
        print(f"Success: {out_path}")
    else:
        print(f"Failed to compile CPU backend for {name}")

def compile_cuda(compiler_info):
    # CUDA usually requires native NVCC. Cross-compiling CUDA is hard.
    # We only build CUDA if host OS matches target OS roughly, or if we are native.
    # For now, simplistic check: only build if "gcc" (Linux) or "cl" (Windows) and nvcc exists.

    if compiler_info["os"] == "windows" and os.name != "nt":
        print("Skipping CUDA cross-compilation (Linux -> Windows not supported yet).")
        return
    if compiler_info["os"] == "linux" and os.name == "nt":
        print("Skipping CUDA cross-compilation (Windows -> Linux via WSL not supported yet for GPU).")
        # WSL2 supports CUDA but invocation is tricky (pass-through).
        return

    if shutil.which("nvcc") is None:
        print("Skipping CUDA: nvcc not found")
        return

    print(f"\n[Build] Compiling CUDA Backend for {compiler_info['name']}...")
    src_dir = PLUGINS_DIR / "cuda" / "c_src"
    src = src_dir / "tensor_ops_cuda.cu"

    targets = [
        {"name": "libtensor_ops_cuda", "flags": []},
        {"name": "libtensor_ops_cuda_sm61", "flags": ["-arch=sm_61"]}
    ]

    for target in targets:
        out_path = src_dir / f"{target['name']}{compiler_info['ext']}"

        # NVCC flags construction
        nvcc_flags = ["-O3", "--shared", "-Xcompiler", "-fPIC"] + target["flags"]
        if compiler_info["os"] == "windows":
             nvcc_flags = ["-O3", "--shared", "-Xcompiler", "/LD"] + target["flags"]

        cmd = ["nvcc", str(src)] + nvcc_flags + ["-o", str(out_path)]
        if run_command(cmd):
            print(f"Success: {out_path}")
        else:
            print(f"Failed: {out_path}")

def compile_opencl(compiler_info, vendor, output_dir):
    print(f"\n[Build] Compiling OpenCL Backend ({vendor}) for {compiler_info['name']}...")
    src = COMMON_C_SRC / "tensor_ops_opencl.c"
    if not src.exists():
        print(f"Warning: {src} not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"libtensor_ops_{vendor.lower()}{compiler_info['ext']}"

    defines = [f"VENDOR_FILTER=\"{vendor}\""]
    libraries = []
    if compiler_info["os"] == "linux":
        libraries.append("dl") # For dlopen

    if compiler_info["compiler"].compile([src], out_path, defines=defines, libraries=libraries):
        print(f"Success: {out_path}")
    else:
        print(f"Failed to compile {vendor} backend")

def main():
    print("Inference-PIO Universal Build Script")
    print("====================================")

    compilers = get_compilers()
    if not compilers:
        print("No suitable compilers found!")
        return

    for c in compilers:
        print(f"\n--- Building for {c['name']} ---")
        compile_cpu(c)
        compile_cuda(c)

        # AMD
        compile_opencl(c, "AMD", PLUGINS_DIR / "amd" / "c_src")
        # Intel
        compile_opencl(c, "Intel", PLUGINS_DIR / "intel" / "c_src")

    print("\nAll builds finished.")

if __name__ == "__main__":
    main()
