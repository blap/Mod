import os
import json
import unittest
from pathlib import Path


class TestInfrastructureConfiguration(unittest.TestCase):
    """Test that the infrastructure configuration has been properly set up."""

    def setUp(self):
        """Set up test fixtures."""
        self.infra_dir = Path("configs/infra")
        self.github_actions_dir = self.infra_dir / "github-actions"
        self.action_templates_dir = self.github_actions_dir / "action-templates"

    def test_infra_directory_structure(self):
        """Test that the infra directory structure exists."""
        self.assertTrue(self.infra_dir.exists(), "configs/infra directory should exist")
        self.assertTrue(self.github_actions_dir.exists(), "configs/infra/github-actions directory should exist")
        self.assertTrue(self.action_templates_dir.exists(), "configs/infra/github-actions/action-templates directory should exist")

    def test_common_env_script_exists(self):
        """Test that the common environment script exists."""
        common_env_script = self.github_actions_dir / "common-env.sh"
        self.assertTrue(common_env_script.exists(), "common-env.sh should exist in github-actions directory")
        
        # Check that it's executable or has proper shebang
        with open(common_env_script, 'r') as f:
            content = f.read()
            self.assertIn("#!/bin/bash", content, "common-env.sh should have bash shebang")

    def test_matrix_strategy_exists(self):
        """Test that the matrix strategy file exists."""
        matrix_file = self.github_actions_dir / "matrix-strategy.json"
        self.assertTrue(matrix_file.exists(), "matrix-strategy.json should exist in github-actions directory")
        
        # Check that it's valid JSON
        with open(matrix_file, 'r') as f:
            try:
                config = json.load(f)
                self.assertIsInstance(config, dict, "matrix-strategy.json should contain a dictionary")
                self.assertIn("python_matrix", config, "matrix-strategy.json should have python_matrix")
                self.assertIn("os_matrix", config, "matrix-strategy.json should have os_matrix")
                self.assertIn("test_scenarios", config, "matrix-strategy.json should have test_scenarios")
            except json.JSONDecodeError:
                self.fail("matrix-strategy.json should be valid JSON")

    def test_action_templates_exist(self):
        """Test that all action templates exist."""
        templates = [
            "build-template.yml",
            "ci-template.yml",
            "deploy-template.yml",
            "lint-template.yml",
            "test-template.yml"
        ]
        
        for template in templates:
            template_path = self.action_templates_dir / template
            self.assertTrue(template_path.exists(), f"{template} should exist in action-templates directory")
            
            # Check that each template has the required structure
            with open(template_path, 'r') as f:
                content = f.read()
                self.assertIn("workflow_call:", content, f"{template} should have workflow_call trigger")

    def test_infra_readme_exists(self):
        """Test that the infrastructure README exists."""
        readme_path = self.infra_dir / "README.md"
        self.assertTrue(readme_path.exists(), "README.md should exist in infra directory")
        
        with open(readme_path, 'r') as f:
            content = f.read()
            self.assertIn("Infrastructure Configuration Guide", content, "README.md should contain infrastructure guide title")

    def test_github_actions_readme_exists(self):
        """Test that the GitHub Actions README exists."""
        readme_path = self.github_actions_dir / "README.md"
        self.assertTrue(readme_path.exists(), "README.md should exist in github-actions directory")
        
        with open(readme_path, 'r') as f:
            content = f.read()
            self.assertIn("GitHub Actions Configuration", content, "README.md should contain GitHub Actions title")

    def test_infra_config_exists(self):
        """Test that the infrastructure config file exists."""
        config_path = self.infra_dir / "config.json"
        self.assertTrue(config_path.exists(), "config.json should exist in infra directory")
        
        with open(config_path, 'r') as f:
            try:
                config = json.load(f)
                self.assertIsInstance(config, dict, "config.json should contain a dictionary")
                self.assertIn("infrastructure_config", config, "config.json should have infrastructure_config")
            except json.JSONDecodeError:
                self.fail("config.json should be valid JSON")

    def test_completion_report_exists(self):
        """Test that the completion report exists."""
        report_path = self.infra_dir / "COMPLETION_REPORT.md"
        self.assertTrue(report_path.exists(), "COMPLETION_REPORT.md should exist in infra directory")
        
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("Configuration Reorganization Completion Report", content, "Completion report should have correct title")


if __name__ == "__main__":
    # Change to the project root directory to resolve relative paths correctly
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    # Run the tests
    unittest.main()