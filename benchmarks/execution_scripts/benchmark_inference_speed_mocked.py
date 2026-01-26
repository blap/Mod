"""
Real Benchmark Executor for Inference Speed - All Models

This script runs inference speed benchmarks for all models without using
pytest or unittest frameworks. It uses the real models located on drive H.
"""

import sys
import time
import torch
from pathlib import Path
import importlib
import traceback
import os
import psutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def load_real_model_plugin(model_name: str):
    """
    Load the real plugin for the specified model from drive H.
    """
    # Define the real model paths on drive H
    model_paths = {
        "glm_4_7": "H:/GLM-4.7",
        "qwen3_4b_instruct_2507": "H:/Qwen3-4B-Instruct-2507",
        "qwen3_coder_30b": "H:/Qwen3-Coder-30B-A3B-Instruct",
        "qwen3_vl_2b": "H:/Qwen3-VL-2B-Instruct"
    }

    if model_name not in model_paths:
        raise ValueError(f"Model {model_name} not found in model paths")

    model_path = model_paths[model_name]

    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    logger.info(f"Loading real model from {model_path} for {model_name}")

    # Import the appropriate model plugin
    try:
        if model_name == "glm_4_7":
            from inference_pio.models.glm_4_7.model import GLM47Model
            plugin = GLM47Model(model_path=model_path)
        elif model_name == "qwen3_4b_instruct_2507":
            from inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
            plugin = Qwen34BInstruct2507Model(model_path=model_path)
        elif model_name == "qwen3_coder_30b":
            from inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
            plugin = Qwen3Coder30BModel(model_path=model_path)
        elif model_name == "qwen3_vl_2b":
            from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
            plugin = Qwen3VL2BModel(model_path=model_path)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Initialize the plugin
        if not plugin.initialize():
            raise RuntimeError(f"Failed to initialize plugin for {model_name}")

        logger.info(f"Successfully loaded and initialized model for {model_name}")
        return plugin
    except ImportError as e:
        logger.error(f"Failed to import model module for {model_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model for {model_name}: {e}")
        raise

def run_inference_speed_benchmark_for_model(model_name: str):
    """
    Run inference speed benchmark for a specific model using real plugin.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Inference Speed Benchmark for {model_name} (Real Model)")
    logger.info(f"{'='*60}")

    try:
        # Load the real model plugin
        plugin = load_real_model_plugin(model_name)

        # Import the benchmark module
        module_path = f"inference_pio.models.{model_name}.benchmarks.performance.benchmark_inference_speed"
        try:
            benchmark_module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error(f"Failed to import {module_path}: {e}")
            return {"error": str(e), "status": "failed"}

        # Find the benchmark class
        benchmark_class = None
        for attr_name in dir(benchmark_module):
            attr = getattr(benchmark_module, attr_name)
            if (hasattr(attr, '__bases__') and
                len(attr.__bases__) > 0 and
                'Benchmark' in attr_name and
                attr_name.endswith('InferenceSpeed')):
                benchmark_class = attr
                break

        if not benchmark_class:
            logger.error(f"No benchmark class found in {module_path}")
            return {"error": "No benchmark class found", "status": "failed"}

        # Create an instance of the benchmark class
        benchmark_instance = benchmark_class()

        # Set the real plugin
        benchmark_instance.plugin = plugin

        # Manually call setUp logic with the real model
        try:
            benchmark_instance.setUp()
            if not hasattr(benchmark_instance, 'model') or benchmark_instance.model is None:
                # If setUp didn't properly initialize, try manual initialization
                benchmark_instance.model = plugin
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return {"error": str(e), "status": "failed"}

        # Run specific inference speed tests
        results = {
            "model": model_name,
            "category": "inference_speed",
            "timestamp": time.time(),
            "results": [],
            "status": "success"
        }

        test_methods = [
            'test_inference_speed_short_input',
            'test_inference_speed_medium_input',
            'test_inference_speed_long_input',
            'test_generation_speed',
            'test_batch_inference_speed',
            'test_variable_length_inference_speed'
        ]

        for method_name in test_methods:
            if hasattr(benchmark_instance, method_name):
                logger.info(f"\nRunning {method_name}...")
                try:
                    method = getattr(benchmark_instance, method_name)

                    # Capture print output
                    import io
                    import contextlib

                    output_buffer = io.StringIO()
                    with contextlib.redirect_stdout(output_buffer):
                        start_time = time.time()
                        method()
                        end_time = time.time()

                    output = output_buffer.getvalue()

                    results["results"].append({
                        "test_method": method_name,
                        "status": "passed",
                        "output": output,
                        "execution_time": end_time - start_time
                    })

                    logger.info(f"✓ {method_name} completed in {end_time - start_time:.2f}s")

                except Exception as e:
                    error_msg = f"Failed to run {method_name}: {str(e)}"
                    logger.error(f"✗ {method_name} failed: {error_msg}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

                    results["results"].append({
                        "test_method": method_name,
                        "status": "failed",
                        "error": error_msg,
                        "traceback": traceback.format_exc()
                    })
                    results["status"] = "partial"
            else:
                logger.warning(f"Method {method_name} not found in benchmark class")

        # Call tearDown if it exists
        if hasattr(benchmark_instance, 'tearDown'):
            try:
                benchmark_instance.tearDown()
            except Exception as e:
                logger.error(f"TearDown failed: {e}")

        # Cleanup the plugin
        try:
            plugin.cleanup()
        except Exception as e:
            logger.warning(f"Plugin cleanup failed: {e}")

        return results

    except Exception as e:
        logger.error(f"Failed to run benchmark for {model_name}: {e}")
        return {"error": str(e), "status": "failed"}

def run_all_inference_speed_benchmarks():
    """
    Run inference speed benchmarks for all models using real plugins.
    """
    models = [
        "glm_4_7",
        "qwen3_4b_instruct_2507",
        "qwen3_coder_30b",
        "qwen3_vl_2b"
    ]

    logger.info("Starting Inference Speed Benchmarks for All Models (Real Models)...")

    all_results = {}
    for model in models:
        logger.info(f"\nProcessing model: {model}")
        result = run_inference_speed_benchmark_for_model(model)
        all_results[model] = result

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE SPEED BENCHMARK SUMMARY (REAL MODELS)")
    logger.info("="*60)

    for model, result in all_results.items():
        if result.get("status") == "success":
            passed = sum(1 for r in result.get("results", []) if r.get("status") == "passed")
            total = len(result.get("results", []))
            logger.info(f"{model}: {passed}/{total} tests passed")
        else:
            logger.info(f"{model}: FAILED - {result.get('error', 'Unknown error')}")

    return all_results

if __name__ == "__main__":
    results = run_all_inference_speed_benchmarks()

    # Save results to file
    import json
    with open("inference_speed_benchmark_results_real.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to inference_speed_benchmark_results_real.json")