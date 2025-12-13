# GitHub Actions Configuration

This directory contains reusable GitHub Actions configurations and templates for the Qwen3-VL project.

## Templates

The templates in `action-templates/` provide standardized workflows that can be reused across different projects:

- `build-template.yml` - Standardized build workflow
- `ci-template.yml` - Continuous Integration workflow template
- `deploy-template.yml` - Deployment workflow template
- `lint-template.yml` - Code quality checking workflow
- `test-template.yml` - Testing workflow template

## Common Configuration Files

- `common-env.sh` - Shell script with common environment setup for GitHub Actions
- `matrix-strategy.json` - Shared matrix strategies for parallel job execution

## Usage

These templates can be referenced in your main workflow files using GitHub Actions' composite action mechanism or by calling them as reusable workflows.