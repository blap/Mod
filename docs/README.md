# Qwen3-VL Documentation

Welcome to the documentation for the Qwen3-VL multimodal model optimization project. This documentation is organized into three main categories to serve different audiences:

## [User Documentation](user/README.md)
Documentation for end users of the Qwen3-VL model, including installation guides, tutorials, and usage instructions.

- [Getting Started](user/guides/getting_started/README.md)
  - [Installation Guide](user/guides/getting_started/installation.md)
  - [Quick Start Guide](user/guides/getting_started/quick_start.md)
- [User Guides](user/guides/README.md)
- [Tutorials](user/tutorials/README.md)

## [Developer Documentation](dev/README.md)
Documentation for developers contributing to or extending the Qwen3-VL model, including architecture, implementation details, and development practices.

- [Architecture Overview](architecture/README.md)
- [Development Guides](dev/guides/README.md)
- [Testing Documentation](dev/testing/README.md)

## [API Documentation](api/README.md)
Technical reference documentation for the Qwen3-VL model APIs, including specifications and implementation details.

- [API Reference](api/reference/README.md)
- [Specifications](api/specifications/README.md)

## [Optimization Guides](dev/architecture/README.md)
Detailed documentation on various optimization techniques implemented in the Qwen3-VL model:

- [Performance Optimization](dev/architecture/performance/)
- [CPU Optimization](dev/architecture/cpu_optimization/)
- [Memory Optimization](dev/architecture/memory_optimization/)
- [SIMD Optimization](dev/architecture/simd_optimization/)
- [System-Level Optimization](dev/architecture/system_optimization/)
- [Power Management](dev/architecture/power_management/)
- [Thread Safety](dev/architecture/thread_safety/)

## Navigation

- [Complete Table of Contents](SUMMARY.md) - Full navigation structure
- [Index](INDEX.md) - Alternative navigation view
- [Changelog](../CHANGELOG.md) - Project release history and changes
- [Contributing Guidelines](../CONTRIBUTING.md) - Information for contributors

## Structure Overview

```
docs/
├── user/                 # User-focused documentation
│   ├── guides/           # User guides
│   │   └── getting_started/  # Getting started materials
│   └── tutorials/        # Tutorials
├── architecture/         # System architecture documentation (consolidated)
├── dev/                  # Developer-focused documentation
│   ├── guides/           # Development guides
│   └── testing/          # Testing documentation
├── api/                  # API reference documentation
│   ├── reference/        # API references
│   └── specifications/   # API specifications
├── SUMMARY.md            # Main table of contents
├── INDEX.md              # Alternative index view
└── README.md             # This file
```

This structure separates concerns by audience and purpose, making it easier for users to find the information most relevant to their needs.