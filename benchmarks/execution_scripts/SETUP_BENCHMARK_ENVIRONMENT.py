#!/usr/bin/env python3
"""
Dependency Installation Script for Comprehensive Benchmark Environment

This script installs all necessary dependencies for running benchmarks
on both original and modified model states.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_package(package):
    """Install a single package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False


def install_requirements(requirements_file="requirements.txt"):
    """Install packages from a requirements file."""
    if not os.path.exists(requirements_file):
        print(f"Requirements file {requirements_file} not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"✓ Successfully installed packages from {requirements_file}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install packages from {requirements_file}")
        return False


def install_additional_benchmark_deps():
    """Install additional dependencies specifically for benchmarking."""
    additional_packages = [
        "pandas>=2.1.4",
        "matplotlib>=3.8.2", 
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "memory-profiler>=0.61.0",
        "line-profiler>=4.1.2",
        "py-cpuinfo>=9.0.0",
        "psutil>=5.9.7",
        "GPUtil>=1.4.0",
        "pynvml>=11.5.0",
        "nvidia-ml-py3>=7.352.0"
    ]
    
    success_count = 0
    for package in additional_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstalled {success_count}/{len(additional_packages)} additional benchmark packages")
    return success_count == len(additional_packages)


def verify_installations():
    """Verify that critical packages are installed."""
    critical_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"), 
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("pandas", "pandas")
    ]
    
    print("\nVerifying critical package installations...")
    success_count = 0
    
    for import_name, display_name in critical_packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name} is available")
            success_count += 1
        except ImportError:
            print(f"✗ {display_name} is NOT available")
    
    print(f"\nVerified {success_count}/{len(critical_packages)} critical packages")
    return success_count == len(critical_packages)


def setup_directories():
    """Create necessary directories for benchmarking."""
    dirs_to_create = [
        "benchmark_results",
        "logs"
    ]

    for directory in dirs_to_create:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def main():
    """Main installation function."""
    print("Inference-PIO Benchmark Environment Setup")
    print("="*50)
    
    print("\nStep 1: Installing core requirements...")
    core_success = install_requirements("requirements.txt")
    
    print("\nStep 2: Installing benchmark-specific requirements...")
    bench_success = install_requirements("requirements_benchmark.txt")
    
    print("\nStep 3: Installing additional benchmark dependencies...")
    additional_success = install_additional_benchmark_deps()
    
    print("\nStep 4: Creating necessary directories...")
    setup_directories()
    
    print("\nStep 5: Verifying installations...")
    verify_success = verify_installations()
    
    print("\n" + "="*50)
    print("INSTALLATION SUMMARY")
    print("="*50)
    print(f"Core requirements: {'✓' if core_success else '✗'}")
    print(f"Benchmark requirements: {'✓' if bench_success else '✗'}")
    print(f"Additional dependencies: {'✓' if additional_success else '✗'}")
    print(f"Package verification: {'✓' if verify_success else '✗'}")
    
    overall_success = all([core_success, bench_success, additional_success, verify_success])
    
    if overall_success:
        print(f"\n🎉 All installations completed successfully!")
        print(f"Environment is ready for comprehensive benchmarking.")
    else:
        print(f"\n⚠️  Some installations failed. Please check the output above.")
        print(f"You may need to install missing packages manually.")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)