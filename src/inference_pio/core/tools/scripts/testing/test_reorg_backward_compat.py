"""
Script to maintain backward compatibility after test reorganization.
This creates symbolic links or copies test files back to their original locations
while preserving the new standardized structure.
"""

import os
import shutil
from pathlib import Path


def create_backward_compatibility():
    """Create backward compatibility by linking/moving files back to original locations."""

    # Define mapping of new location to old location
    test_mappings = [
        # GLM-4-7-flash model tests
        (
            "tests/unit/models/test_glm47_attention.py",
            "src/inference_pio/models/glm_4_7_flash/tests/unit/test_attention.py",
        ),
        # GLM-4-7-flash integration tests
        (
            "tests/integration/models/test_glm47_config_loading.py",
            "src/inference_pio/models/glm_4_7_flash/tests/integration/test_config_loading.py",
        ),
        (
            "tests/integration/models/test_glm47_model_loading.py",
            "src/inference_pio/models/glm_4_7_flash/tests/integration/test_model_loading.py",
        ),
        (
            "tests/integration/models/test_glm47_end_to_end.py",
            "src/inference_pio/models/glm_4_7_flash/tests/integration/test_end_to_end.py",
        ),
        (
            "tests/integration/models/test_glm47_inference.py",
            "src/inference_pio/models/glm_4_7_flash/tests/integration/test_inference.py",
        ),
        (
            "tests/integration/models/test_glm47_optimized_inference.py",
            "src/inference_pio/models/glm_4_7_flash/tests/integration/test_optimized_inference.py",
        ),
        (
            "tests/integration/models/test_glm47_plugin_integration.py",
            "src/inference_pio/models/glm_4_7_flash/tests/integration/test_plugin_integration.py",
        ),
        # GLM-4-7-flash performance tests
        (
            "tests/performance/models/test_glm47_specific_optimizations.py",
            "src/inference_pio/models/glm_4_7_flash/tests/performance/test_glm47_specific_optimizations.py",
        ),
        (
            "tests/performance/models/test_glm47_paged_attention.py",
            "src/inference_pio/models/glm_4_7_flash/tests/performance/test_paged_attention.py",
        ),
        # Qwen3-4b-instruct-2507 model tests
        (
            "tests/unit/models/test_qwen3_4b_attention.py",
            "src/inference_pio/models/qwen3_4b_instruct_2507/tests/unit/test_attention.py",
        ),
        (
            "tests/unit/models/test_qwen3_4b_unit_optimizations.py",
            "src/inference_pio/models/qwen3_4b_instruct_2507/tests/unit/test_qwen3_unit_optimizations.py",
        ),
        # Qwen3-4b-instruct-2507 integration tests
        (
            "tests/integration/models/test_qwen3_4b_config_loading.py",
            "src/inference_pio/models/qwen3_4b_instruct_2507/tests/integration/test_config_loading.py",
        ),
        (
            "tests/integration/models/test_qwen3_4b_model_loading.py",
            "src/inference_pio/models/qwen3_4b_instruct_2507/tests/integration/test_model_loading.py",
        ),
        (
            "tests/integration/models/test_qwen3_4b_end_to_end.py",
            "src/inference_pio/models/qwen3_4b_instruct_2507/tests/integration/test_end_to_end.py",
        ),
        # Qwen3-4b-instruct-2507 performance tests
        (
            "tests/performance/models/test_qwen3_4b_standard_plugin_interface.py",
            "src/inference_pio/models/qwen3_4b_instruct_2507/tests/performance/test_standard_plugin_interface.py",
        ),
        # Qwen3-coder-30b model tests
        (
            "tests/unit/models/test_qwen3_coder_attention.py",
            "src/inference_pio/models/qwen3_coder_30b/tests/unit/test_attention.py",
        ),
        # Qwen3-coder-30b integration tests
        (
            "tests/integration/models/test_qwen3_coder_config_loading.py",
            "src/inference_pio/models/qwen3_coder_30b/tests/integration/test_config_loading.py",
        ),
        (
            "tests/integration/models/test_qwen3_coder_model_loading.py",
            "src/inference_pio/models/qwen3_coder_30b/tests/integration/test_model_loading.py",
        ),
        (
            "tests/integration/models/test_qwen3_coder_end_to_end.py",
            "src/inference_pio/models/qwen3_coder_30b/tests/integration/test_end_to_end.py",
        ),
        # Qwen3-vl-2b model tests
        (
            "tests/unit/models/test_qwen3_vl_attention.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/unit/test_attention.py",
        ),
        (
            "tests/unit/models/test_qwen3_vl_async_multimodal_processing_unit.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/unit/test_async_multimodal_processing_unit.py",
        ),
        # Qwen3-vl-2b integration tests
        (
            "tests/integration/models/test_qwen3_vl_config_loading.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/integration/test_config_loading.py",
        ),
        (
            "tests/integration/models/test_qwen3_vl_model_loading.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/integration/test_model_loading.py",
        ),
        (
            "tests/integration/models/test_qwen3_vl_end_to_end.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/integration/test_end_to_end.py",
        ),
        # Qwen3-vl-2b performance tests
        (
            "tests/performance/models/test_qwen3_vl_cuda_kernels.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/performance/test_qwen3_vl_cuda_kernels.py",
        ),
        (
            "tests/performance/models/test_qwen3_vl_vision_transformer_kernels.py",
            "src/inference_pio/models/qwen3_vl_2b/tests/performance/test_vision_transformer_kernels.py",
        ),
        # Common tests
        (
            "tests/unit/common/test_contractual_interfaces.py",
            "src/inference_pio/common/tests/test_contractual_interfaces.py",
        ),
        (
            "tests/unit/common/test_model_surgery.py",
            "src/inference_pio/common/tests/test_model_surgery.py",
        ),
        (
            "tests/unit/common/test_multimodal_attention.py",
            "src/inference_pio/common/tests/test_multimodal_attention.py",
        ),
        (
            "tests/unit/common/test_quantization.py",
            "src/inference_pio/common/tests/test_quantization.py",
        ),
        (
            "tests/unit/common/test_optimization_profiles.py",
            "src/inference_pio/common/tests/test_optimization_profiles.py",
        ),
        (
            "tests/unit/common/test_feedback_controller.py",
            "src/inference_pio/common/tests/test_feedback_controller.py",
        ),
        (
            "tests/unit/common/test_input_complexity_analyzer.py",
            "src/inference_pio/common/tests/test_input_complexity_analyzer.py",
        ),
        (
            "tests/unit/common/test_structured_pruning.py",
            "src/inference_pio/common/tests/test_structured_pruning.py",
        ),
        (
            "tests/unit/common/test_tensor_decomposition.py",
            "src/inference_pio/common/tests/test_tensor_decomposition.py",
        ),
        # Common integration tests
        (
            "tests/integration/common/test_adaptive_batch_manager_complexity.py",
            "src/inference_pio/common/tests/test_adaptive_batch_manager_complexity.py",
        ),
        (
            "tests/integration/common/test_disk_offloading.py",
            "src/inference_pio/common/tests/test_disk_offloading.py",
        ),
        (
            "tests/integration/common/test_dynamic_config_system.py",
            "src/inference_pio/common/tests/test_dynamic_config_system.py",
        ),
        (
            "tests/integration/common/test_model_surgery_integration.py",
            "src/inference_pio/common/tests/test_model_surgery_integration.py",
        ),
        (
            "tests/integration/common/test_multimodal_model_surgery.py",
            "src/inference_pio/common/tests/test_multimodal_model_surgery.py",
        ),
        (
            "tests/integration/common/test_pipeline_parallel.py",
            "src/inference_pio/common/tests/test_pipeline_parallel.py",
        ),
        (
            "tests/integration/common/test_sequence_parallel.py",
            "src/inference_pio/common/tests/test_sequence_parallel.py",
        ),
        (
            "tests/integration/common/test_tensor_pagination.py",
            "src/inference_pio/common/tests/test_tensor_pagination.py",
        ),
        (
            "tests/integration/common/test_optimization_profiles_extended.py",
            "src/inference_pio/common/tests/test_optimization_profiles_extended.py",
        ),
        (
            "tests/integration/common/test_ml_optimization_system.py",
            "src/inference_pio/common/tests/test_ml_optimization_system.py",
        ),
        (
            "tests/integration/common/test_streaming_computation.py",
            "src/inference_pio/common/tests/test_streaming_computation.py",
        ),
        (
            "tests/integration/common/test_unimodal_model_surgery.py",
            "src/inference_pio/common/tests/test_unimodal_model_surgery.py",
        ),
        (
            "tests/integration/common/test_unimodal_tensor_pagination.py",
            "src/inference_pio/common/tests/test_unimodal_tensor_pagination.py",
        ),
        (
            "tests/integration/common/test_vision_language_parallel.py",
            "src/inference_pio/common/tests/test_vision_language_parallel.py",
        ),
        (
            "tests/integration/common/test_security_isolation_extended.py",
            "src/inference_pio/common/tests/test_security_isolation_extended.py",
        ),
        # Basic tests from original inference_pio tests
        ("tests/unit/common/basic_tests.py", "src/inference_pio/tests/basic_tests.py"),
        ("tests/unit/common/quick_test.py", "src/inference_pio/tests/quick_test.py"),
        (
            "tests/unit/common/simple_config_test.py",
            "src/inference_pio/tests/simple_config_test.py",
        ),
        (
            "tests/unit/common/simple_test_verification.py",
            "src/inference_pio/tests/simple_test_verification.py",
        ),
        (
            "tests/unit/common/test_sample_discovery.py",
            "src/inference_pio/tests/test_sample_discovery.py",
        ),
        (
            "tests/unit/common/test_test_discovery.py",
            "src/inference_pio/tests/test_test_discovery.py",
        ),
        (
            "tests/unit/common/verify_documentation_standardization.py",
            "src/inference_pio/tests/verify_documentation_standardization.py",
        ),
        (
            "tests/integration/common/demonstrate_dynamic_text_batching.py",
            "src/inference_pio/tests/demonstrate_dynamic_text_batching.py",
        ),
        (
            "tests/integration/common/final_test.py",
            "src/inference_pio/tests/final_test.py",
        ),
        (
            "tests/integration/common/test_pagination_system.py",
            "src/inference_pio/tests/test_pagination_system.py",
        ),
    ]

    # Create directories if they don't exist
    for _, old_location in test_mappings:
        old_path = Path(old_location)
        old_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy files back to original locations to maintain backward compatibility
    for new_location, old_location in test_mappings:
        new_path = Path(new_location)
        old_path = Path(old_location)

        if new_path.exists():
            try:
                # Copy the file back to its original location
                shutil.copy2(new_path, old_path)
                print(f"Copied {new_path} back to {old_path}")
            except Exception as e:
                print(f"Failed to copy {new_path} to {old_path}: {e}")
        else:
            print(f"Warning: New location {new_path} does not exist")


def create_init_files():
    """Create __init__.py files in test directories to make them importable."""

    test_dirs = [
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "tests/unit/models",
        "tests/unit/common",
        "tests/integration/models",
        "tests/integration/common",
        "tests/performance/models",
        "tests/performance/common",
        "src/inference_pio/tests",
        "src/inference_pio/models/glm_4_7_flash/tests/unit",
        "src/inference_pio/models/glm_4_7_flash/tests/integration",
        "src/inference_pio/models/glm_4_7_flash/tests/performance",
        "src/inference_pio/models/qwen3_4b_instruct_2507/tests/unit",
        "src/inference_pio/models/qwen3_4b_instruct_2507/tests/integration",
        "src/inference_pio/models/qwen3_4b_instruct_2507/tests/performance",
        "src/inference_pio/models/qwen3_coder_30b/tests/unit",
        "src/inference_pio/models/qwen3_coder_30b/tests/integration",
        "src/inference_pio/models/qwen3_coder_30b/tests/performance",
        "src/inference_pio/models/qwen3_vl_2b/tests/unit",
        "src/inference_pio/models/qwen3_vl_2b/tests/integration",
        "src/inference_pio/models/qwen3_vl_2b/tests/performance",
        "src/inference_pio/common/tests",
    ]

    for test_dir in test_dirs:
        init_file = Path(test_dir) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")


if __name__ == "__main__":
    print("Creating backward compatibility for reorganized tests...")
    create_backward_compatibility()
    create_init_files()
    print("Backward compatibility setup completed.")
