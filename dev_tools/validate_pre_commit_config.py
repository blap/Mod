#!/usr/bin/env python3
"""
Validation script for pre-commit configuration.
This script validates that the pre-commit configuration is properly formatted
and includes all required hooks for the Qwen3-VL project.
"""

import os
import sys
import yaml
from pathlib import Path


def validate_pre_commit_config(config_path: str):
    """Validate the pre-commit configuration file."""
    path = Path(config_path)
    
    if not path.exists():
        print(f"ERROR: Pre-commit config file does not exist: {config_path}")
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in {config_path}: {e}")
        return False
    
    # Validate basic structure
    if 'repos' not in config:
        print(f"ERROR: Missing 'repos' key in {config_path}")
        return False
    
    if not isinstance(config['repos'], list):
        print(f"ERROR: 'repos' should be a list in {config_path}")
        return False
    
    # Check for required hooks
    required_hooks = {'black', 'isort', 'flake8', 'mypy'}
    found_hooks = set()
    
    for repo in config['repos']:
        if 'hooks' not in repo:
            print(f"ERROR: Missing 'hooks' in repo {repo.get('repo', 'unknown')} in {config_path}")
            return False
        
        for hook in repo['hooks']:
            if 'id' in hook:
                found_hooks.add(hook['id'])
    
    missing_hooks = required_hooks - found_hooks
    if missing_hooks:
        print(f"ERROR: Missing required hooks in {config_path}: {missing_hooks}")
        return False
    
    print(f"SUCCESS: {config_path} is valid and contains all required hooks")
    return True


def main():
    """Main function to validate pre-commit configurations."""
    print("Validating pre-commit configurations...")
    
    configs_to_check = [
        '.pre-commit-config.yaml',  # Original
        '.pre-commit-config-recommended.yaml'  # Recommended
    ]
    
    all_valid = True
    
    for config in configs_to_check:
        if not validate_pre_commit_config(config):
            all_valid = False
    
    if all_valid:
        print("\nAll pre-commit configurations are valid!")
        return 0
    else:
        print("\nSome pre-commit configurations have issues!")
        return 1


if __name__ == "__main__":
    sys.exit(main())