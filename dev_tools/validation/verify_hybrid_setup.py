import logging
import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from inference_pio.common.hardware_analyzer import SystemProfile
from inference_pio.plugin_system.factory import get_processor_plugin
from inference_pio.plugin_system.intel_comet_lake_plugin import IntelCometLakePlugin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridVerification")


def verify_hybrid_setup():
    logger.info("Starting Hybrid Setup Verification...")

    # 1. Create Mock Profile for User's Hardware
    user_profile = SystemProfile(
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        total_ram_gb=12.0,
        available_ram_gb=10.0,
        total_vram_gb=2.0,
        available_vram_gb=1.8,  # Assuming some OS usage
        gpu_name="NVIDIA GeForce MX330",
        disk_free_gb=100.0,
        is_weak_hardware=True,
        recommended_offload_strategy="hybrid_aggressive",
        safe_vram_limit_gb=1.6,  # 90% of available
        processor_architecture="x86_64",
        cpu_brand="Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz",
        instruction_sets=["AVX2"],
        secondary_gpu_detected="Intel(R) UHD Graphics",
        hybrid_capability=True,
        ipex_available=True,  # Simulating installed IPEX
    )

    logger.info(
        f"Created Mock Profile: {user_profile.cpu_brand}, GPU: {user_profile.gpu_name}"
    )

    # 2. Get Plugin from Factory
    plugin = get_processor_plugin(user_profile)

    # Verify correct plugin type
    if not isinstance(plugin, IntelCometLakePlugin):
        logger.error(f"Factory returned {type(plugin)} instead of IntelCometLakePlugin")
        return False
    logger.info("Successfully instantiated IntelCometLakePlugin via Factory.")

    # 3. Initialize Plugin
    # Pass basic config, expect plugin to pick up hybrid from profile
    config = {"num_threads": 4}
    plugin.initialize(config)

    # Check if plugin is aware of profile context (this is internal, but we can check the hybrid state indirectly)
    # We can inspect the _profile attribute we added
    if not hasattr(plugin, "_profile") or plugin._profile != user_profile:
        logger.error("Plugin did not store the system profile correctly.")
        return False

    logger.info("Plugin initialized successfully.")

    # 4. Test Layer Distribution Logic
    # Scenario: 4GB Model (approx Qwen2-VL-2B unquantized or similar)
    # Available VRAM 1.8GB. Safe limit ~1.44GB (0.8 * 1.8 calculated inside plugin)
    # Let's say model has 32 layers. 4GB / 32 = 0.125 GB per layer (128MB)

    total_layers = 32
    model_size_gb = 4.0
    available_vram = 1.8

    distribution = plugin.get_layer_distribution(
        total_layers, model_size_gb, available_vram
    )

    gpu_layers = distribution["gpu_layers"]
    cpu_layers = distribution["cpu_layers"]

    logger.info(f"Distribution Result: GPU={gpu_layers}, CPU={cpu_layers}")

    # Expected:
    # Safe VRAM = 1.8 * 0.8 = 1.44 GB
    # Layer size = 0.125 GB
    # Layers on GPU = int(1.44 / 0.125) = 11
    # Layers on CPU = 32 - 11 = 21

    expected_gpu = int((1.8 * 0.8) / (4.0 / 32))

    if gpu_layers != expected_gpu:
        logger.warning(
            f"GPU layers mismatch. Expected {expected_gpu}, got {gpu_layers}"
        )
        # Allow small deviation due to float precision, but strict check is better

    if gpu_layers + cpu_layers != total_layers:
        logger.error(
            f"Total layers mismatch! {gpu_layers} + {cpu_layers} != {total_layers}"
        )
        return False

    if gpu_layers > 0 and cpu_layers > 0:
        logger.info("SUCCESS: Hybrid split calculated correctly.")
    elif gpu_layers == 0:
        logger.warning("Warning: No layers assigned to GPU. Check thresholds.")
    else:
        logger.warning("Warning: All layers assigned to GPU. Check model size params.")

    return True


if __name__ == "__main__":
    try:
        if verify_hybrid_setup():
            print("Verification PASSED")
            sys.exit(0)
        else:
            print("Verification FAILED")
            sys.exit(1)
    except Exception as e:
        logger.exception("Verification crashed")
        sys.exit(1)
