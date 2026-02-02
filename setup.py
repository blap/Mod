"""
Setup configuration for Inference-PIO package

Inference-PIO: Self-Contained Plugin Architecture for Advanced Model Inference
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from setuptools import find_packages, setup

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt if it exists
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.14.0",
        "sentencepiece",
        "numpy>=1.21.0",
        "Pillow",
        "accelerate",
        "datasets",
        "evaluate",
        "scipy",
        "safetensors",
        "flash-attn>=2.4.0",
        "xformers>=0.0.23",
        "einops>=0.7.0",
    ]

setup(
    name="inference-pio",
    version="1.0.0",
    author="Inference-PIO Team",
    author_email="inference-pio@maintainers.com",
    description="Inference-PIO: Self-Contained Plugin Architecture for Advanced Model Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inference-pio/inference-pio",
    packages=find_packages(
        where="src",
        include=[
            "inference_pio",
            "inference_pio.*",
            "models",
            "models.*",
            "plugins",
            "plugins.*",
            "common",
            "common.*",
            "inference",
            "inference.*",
            "utils",
            "utils.*",
        ],
    ),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
        ],
        "training": [
            "deepspeed",
            "fairscale",
            "torch-tensorrt",
        ],
        "inference": [
            "onnx",
            "onnxruntime",
            "tensorrt",
        ],
        "cuda": [
            "ninja",
            "pybind11",
        ],
        "vision": [
            "Pillow",
            "opencv-python",
            "timm",
        ],
        "multimodal": [
            "Pillow",
            "opencv-python",
            "timm",
        ],
    },
    entry_points={
        "console_scripts": [
            "inference-pio=inference_pio.__main__:main",
        ],
    },
)
