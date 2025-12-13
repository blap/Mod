# Source Code Directory

This directory contains the main source code for the project, organized into functional components.

## Contents

- **components/**: Core components and modules
- **cuda_kernels/**: CUDA kernel implementations for GPU acceleration
- **language/**: Language model components and processing
- **models/**: Model implementations and architectures
- **multimodal/**: Multimodal processing components
- **qwen3_vl/**: Qwen3 Vision-Language specific implementations
- **utils/**: Utility functions and helpers
- **vision/**: Vision processing components

## Purpose

The src directory contains all the core implementation code for the project. Each subdirectory represents a logical component of the system, organized by functionality to promote modularity and maintainability.

## Architecture

The source code follows a modular architecture where each component is responsible for specific functionality:

- **components**: Contains reusable components that may be shared across different parts of the system
- **cuda_kernels**: Implements performance-critical operations using CUDA for GPU acceleration
- **language**: Handles language model processing and text operations
- **models**: Contains model definitions and core architectures
- **multimodal**: Integrates different modalities (text, image, etc.) in a unified framework
- **utils**: Provides common utilities, helpers, and shared functionality
- **vision**: Handles image and visual processing components

## Development

When adding new functionality, please follow the existing directory structure and place code in the most appropriate component directory. Each component should maintain clear interfaces and minimal dependencies on other components.