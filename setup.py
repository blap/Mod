import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path

# This setup.py delegates the actual compilation to build_ops.py
# to maintain the unified build system logic.

class CustomBuildExt(build_ext):
    def run(self):
        # Run our custom build script
        print("[setup.py] invoking build_ops.py...")
        subprocess.check_call([sys.executable, "build_ops.py"])
        # After building, the artifacts are in place.
        # Setuptools will package them if they are in package_data.

setup(
    name="inference_pio",
    version="0.1.0",
    description="High-performance, dependency-free Inference Engine for Qwen/GLM models",
    author="Inference-PIO Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    # Ensure binary artifacts are included
    package_data={
        "inference_pio": [
            "plugins/**/*.dll",
            "plugins/**/*.so",
            "plugins/**/*.dylib",
            "**/*.json", # Configs
            "**/*.txt"   # Vocabs/Merges
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent", # Technically specific binaries, but packaged as wheel
    ],
    python_requires=">=3.8",
    cmdclass={
        'build_ext': CustomBuildExt,
    },
    # Dummy extension to force build_ext to run
    ext_modules=[Extension("inference_pio.dummy", sources=[])],
)
