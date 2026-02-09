"""
Setup configuration for Inference-PIO package

Inference-PIO: Self-Contained Plugin Architecture for Advanced Model Inference
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
Runs on a custom C/CUDA backend (libtensor_ops) without external ML frameworks.
"""

from setuptools import find_packages, setup

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inference-pio",
    version="1.0.0",
    author="Inference-PIO Team",
    author_email="inference-pio@maintainers.com",
    description="Inference-PIO: Self-Contained Plugin Architecture for Advanced Model Inference (Custom C/CUDA Backend)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inference-pio/inference-pio",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "rich>=13.0.0",
        "psutil>=5.9.0",
        "pillow>=10.0.0",  # For image processing in VL models
    ],
    entry_points={
        "console_scripts": [
            "inference-pio=inference_pio.__main__:main",
        ],
    },
)
