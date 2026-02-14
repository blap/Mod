import os
import sys
import subprocess
import shutil
from pathlib import Path

# Configuration
SRC_DIR = Path("src/inference_pio")
PLUGINS_DIR = SRC_DIR / "plugins"
COMMON_C_SRC = PLUGINS_DIR / "common" / "c_src"

# Compiler flags
CFLAGS = ["-O3", "-fPIC", "-shared", "-Wall", "-Wextra", "-fopenmp"]
if os.name == "nt":
    CFLAGS = ["/O2", "/LD", "/openmp"] # MSVC flags

def run_command(cmd, cwd=None):
    """Run a shell command and print output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
        return False

def compile_cpu():
    """Compile CPU backend."""
    print("\n[Build] Compiling CPU Backend...")
    src = PLUGINS_DIR / "cpu" / "c_src" / "tensor_ops.c"
    # Also include the safetensors_loader.c
    loader_src = PLUGINS_DIR / "cpu" / "c_src" / "safetensors_loader.c"

    if not src.exists():
        print(f"Skipping CPU: {src} not found")
        return

    out_name = "libtensor_ops.dll" if os.name == "nt" else "libtensor_ops.so"
    out_path = src.parent / out_name

    cmd = ["gcc"] + [str(src)] + CFLAGS

    # Check if loader exists and append
    if loader_src.exists():
        cmd.append(str(loader_src))

    cmd += ["-o", str(out_path)]

    # Link with math library on Linux
    if os.name != "nt":
        cmd.append("-lm")
        cmd.append("-fopenmp") # Link OpenMP

    if os.name == "nt":
        # MSVC syntax is different - NOT UPDATED FOR LOADER YET (TODO)
        cmd = ["cl", str(src)] + CFLAGS + ["/Fe:" + str(out_path)]

    if run_command(cmd):
        print(f"Success: {out_path}")
    else:
        print("Failed to compile CPU backend.")

def compile_cuda():
    """Compile CUDA backend."""
    print("\n[Build] Compiling CUDA Backend...")
    src_dir = PLUGINS_DIR / "cuda" / "c_src"
    src = src_dir / "tensor_ops_cuda.cu"

    if not src.exists():
        print(f"Skipping CUDA: {src} not found")
        return

    # Check for nvcc
    if shutil.which("nvcc") is None:
        print("Skipping CUDA: nvcc not found in PATH")
        return

    # Define targets
    targets = [
        {"name": "libtensor_ops_cuda", "flags": []}, # Default
        {"name": "libtensor_ops_cuda_sm61", "flags": ["-arch=sm_61"]} # Pascal GTX 10-series
    ]

    for target in targets:
        out_name = f"{target['name']}.dll" if os.name == "nt" else f"{target['name']}.so"
        out_path = src_dir / out_name

        # Basic NVCC flags
        nvcc_flags = ["-O3", "--shared", "-Xcompiler", "-fPIC"] + target["flags"]
        if os.name == "nt":
            nvcc_flags = ["-O3", "--shared", "-Xcompiler", "/LD"] + target["flags"]

        # Compile consolidated source
        sources = [str(src)]

        print(f"Building {out_name}...")
        cmd = ["nvcc"] + sources + nvcc_flags + ["-o", str(out_path)]

        if run_command(cmd):
            print(f"Success: {out_path}")
        else:
            print(f"Failed to compile {out_name}")

def compile_opencl(vendor, output_dir):
    """Compile OpenCL backend for specific vendor."""
    print(f"\n[Build] Compiling OpenCL Backend for {vendor}...")

    # We use the shared common source
    src = COMMON_C_SRC / "tensor_ops_opencl.c"

    # Check if we have the shared source, if not, fallback or warn
    if not src.exists():
        # Fallback to existing AMD one if it exists and we are compiling AMD?
        # But per plan we want to standardize.
        # Let's check if the file exists. If not, maybe we haven't created it yet.
        # For this script to work, the file must exist.
        # Since I'm creating the script first, I'll add a check.
        print(f"Warning: {src} not found. Skipping OpenCL build.")
        return

    out_name = f"libtensor_ops_{vendor.lower()}.dll" if os.name == "nt" else f"libtensor_ops_{vendor.lower()}.so"
    out_path = output_dir / out_name

    # We define VENDOR_FILTER macro
    macros = [f"-DVENDOR_FILTER=\"{vendor}\""]
    if os.name == "nt":
        macros = [f"/DVENDOR_FILTER=\"{vendor}\""]

    cmd = ["gcc", str(src)] + CFLAGS + macros + ["-o", str(out_path)]
    if os.name != "nt":
        cmd.append("-ldl") # For dlopen

    if os.name == "nt":
         cmd = ["cl", str(src)] + CFLAGS + macros + ["/Fe:" + str(out_path)]

    if run_command(cmd):
        print(f"Success: {out_path}")
    else:
        print(f"Failed to compile {vendor} backend.")

def main():
    print("Inference-PIO Build Script")
    print("==========================")

    # 1. CPU
    compile_cpu()

    # 2. CUDA
    compile_cuda()

    # 3. AMD (OpenCL)
    amd_dir = PLUGINS_DIR / "amd" / "c_src"
    amd_dir.mkdir(parents=True, exist_ok=True)
    compile_opencl("AMD", amd_dir)

    # 4. Intel (OpenCL)
    intel_dir = PLUGINS_DIR / "intel" / "c_src"
    intel_dir.mkdir(parents=True, exist_ok=True)
    compile_opencl("Intel", intel_dir)

    print("\nBuild Complete.")

if __name__ == "__main__":
    main()
