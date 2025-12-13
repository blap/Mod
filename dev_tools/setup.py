"""
Setup script for Qwen3-VL Developer Experience Tools
"""

from setuptools import setup, find_packages

# Read the requirements from the requirements file
def read_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name="qwen3-vl-dev-tools",
    version="1.0.0",
    author="Qwen3-VL Development Team",
    author_email="dev@qwen3-vl.example.com",
    description="Developer experience enhancement tools for Qwen3-VL model development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/qwen3-vl/qwen3-vl-dev-tools",
    packages=find_packages(include=['dev_tools', 'dev_tools.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.4.0",
        "pyyaml>=6.0",
        "jinja2>=3.1.0",
        "pygments>=2.11.0",
        "markdown>=3.3.0",
        "psutil>=5.8.0",
        "gputil>=1.4.0",
        "astor>=0.8.1",
    ],
    entry_points={
        'console_scripts': [
            'qwen3-vl-dev = dev_tools:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)