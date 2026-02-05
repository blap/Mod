"""
Configuration file for Qwen3 Coder Next model integration tests.
This file contains configurations needed for integration testing of the model.
"""

# Model-specific configurations
MODEL_NAME = "qwen3_coder_next"
MODEL_PATH = "./models/coding/qwen3_coder_next/"
CONFIG_PATH = "./configs/"

# Integration test configurations
INTEGRATION_TESTS_ENABLED = True
PERFORMANCE_TESTS_ENABLED = True
UNIT_TESTS_ENABLED = True

# Benchmark configurations
BENCHMARK_UNIT_TESTS_ENABLED = True
BENCHMARK_INTEGRATION_TESTS_ENABLED = True
BENCHMARK_PERFORMANCE_TESTS_ENABLED = True