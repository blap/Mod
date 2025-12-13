# Infrastructure Configuration Guide

This directory contains infrastructure-related configurations for the Qwen3-VL project.

## Directory Structure

- `github-actions/` - GitHub Actions reusable configurations and templates
  - `action-templates/` - Template actions that can be reused across workflows
  - `common-env.sh` - Common environment setup for GitHub Actions
  - `matrix-strategy.json` - Shared matrix strategies for parallel jobs

## GitHub Actions Templates

These templates provide standardized workflows that can be reused across different projects:

- `build-template.yml` - Standardized build workflow
- `ci-template.yml` - Continuous Integration workflow template
- `deploy-template.yml` - Deployment workflow template
- `lint-template.yml` - Code quality checking workflow
- `test-template.yml` - Testing workflow template

## Usage

These templates can be called from your main workflow files using the `workflow_call` trigger.