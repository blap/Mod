"""
Test suite for pre-commit configuration validation.
This test ensures the .pre-commit-config.yaml file is properly configured
and validates that it should remain in the root directory as standard practice.
"""
import os
import yaml
from pathlib import Path


def test_pre_commit_config_exists():
    """Test that the pre-commit configuration file exists in the root directory."""
    config_path = Path(".pre-commit-config.yaml")
    assert config_path.exists(), "Pre-commit configuration file should exist in root directory"
    assert config_path.is_file(), "Pre-commit configuration should be a file"


def test_pre_commit_config_content():
    """Test that the pre-commit configuration has expected content."""
    with open(".pre-commit-config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Validate the structure of the configuration
    assert "repos" in config, "Configuration must contain 'repos' key"
    assert isinstance(config["repos"], list), "Repos must be a list"
    
    expected_hooks = {"black", "isort", "flake8", "mypy"}
    found_hooks = set()
    
    for repo in config["repos"]:
        assert "repo" in repo, "Each repo must have a 'repo' key"
        assert "rev" in repo, "Each repo must have a 'rev' key"
        assert "hooks" in repo, "Each repo must have a 'hooks' key"
        
        for hook in repo["hooks"]:
            if "id" in hook:
                found_hooks.add(hook["id"])
    
    # Check that all expected hooks are present
    assert expected_hooks.issubset(found_hooks), f"Missing hooks: {expected_hooks - found_hooks}"


def test_pre_commit_config_location_standard():
    """Test that pre-commit config is in the standard location (root directory)."""
    # According to pre-commit documentation, the config should be in the root
    # of the repository as .pre-commit-config.yaml
    root_config_path = Path(".pre-commit-config.yaml")
    assert root_config_path.exists(), "Pre-commit config should be in repository root"
    
    # Check that it's not in any subdirectory
    for subdir in ["config", "configs", "src", "tests", "dev_tools"]:
        sub_config_path = Path(subdir) / ".pre-commit-config.yaml"
        assert not sub_config_path.exists(), f"Pre-commit config should not be in {subdir} subdirectory"


def test_pre_commit_config_valid_yaml():
    """Test that the pre-commit configuration is valid YAML."""
    with open(".pre-commit-config.yaml", "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            assert config is not None, "YAML should parse to a non-None value"
        except yaml.YAMLError as e:
            raise AssertionError(f"Invalid YAML in pre-commit config: {e}")


def test_expected_tools_configured():
    """Test that expected code quality tools are configured."""
    with open(".pre-commit-config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Find the specific tools
    has_black = any(
        hook.get("id") == "black" 
        for repo in config["repos"] 
        for hook in repo["hooks"]
    )
    
    has_isort = any(
        hook.get("id") == "isort" 
        for repo in config["repos"] 
        for hook in repo["hooks"]
    )
    
    has_flake8 = any(
        hook.get("id") == "flake8" 
        for repo in config["repos"] 
        for hook in repo["hooks"]
    )
    
    has_mypy = any(
        hook.get("id") == "mypy" 
        for repo in config["repos"] 
        for hook in repo["hooks"]
    )
    
    assert has_black, "Black formatter should be configured"
    assert has_isort, "Isort should be configured"
    assert has_flake8, "Flake8 linter should be configured"
    assert has_mypy, "Mypy type checker should be configured"


if __name__ == "__main__":
    # Run the tests manually if executed directly
    test_pre_commit_config_exists()
    print("PASS: Pre-commit config exists test passed")

    test_pre_commit_config_content()
    print("PASS: Pre-commit config content test passed")

    test_pre_commit_config_location_standard()
    print("PASS: Pre-commit config location standard test passed")

    test_pre_commit_config_valid_yaml()
    print("PASS: Pre-commit config valid YAML test passed")

    test_expected_tools_configured()
    print("PASS: Expected tools configured test passed")

    print("\nAll tests passed! The pre-commit configuration is valid and correctly located.")