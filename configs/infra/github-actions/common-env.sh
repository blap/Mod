#!/bin/bash
# Common environment setup for GitHub Actions workflows

# Set common environment variables
export PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
export NODE_VERSION="${NODE_VERSION:-18}"

# Setup common paths
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG_PATH="$PROJECT_ROOT/configs"

# Common functions for GitHub Actions
setup_python() {
    if command -v pyenv &> /dev/null; then
        pyenv install "$PYTHON_VERSION" || true
        pyenv local "$PYTHON_VERSION"
    fi
    python -m pip install --upgrade pip setuptools
}

setup_node() {
    if command -v nvm &> /dev/null; then
        nvm install "$NODE_VERSION"
        nvm use "$NODE_VERSION"
    fi
}

# Print environment info for debugging
print_env_info() {
    echo "Environment setup completed"
    echo "Python version: $(python --version 2>/dev/null || echo 'not installed')"
    echo "Node version: $(node --version 2>/dev/null || echo 'not installed')"
    echo "Project root: $PROJECT_ROOT"
}