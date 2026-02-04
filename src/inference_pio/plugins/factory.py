"""
Plugin Factory for Automatic Processor Plugin Selection

This module provides the factory logic to automatically select and
instantiate the most appropriate hardware processor plugin based on
the system profile.
"""

import logging
from typing import Optional

from ..common.hardware.hardware_analyzer import (
    SystemProfile,
    get_system_profile,
)
from .base.plugin_interface import HardwareProcessorPluginInterface
from .cpu.cpu_plugin import create_generic_cpu_plugin
from .intel.intel_comet_lake_plugin import create_intel_comet_lake_plugin
from .intel.intel_kaby_lake_plugin import create_intel_kaby_lake_plugin
from .amd.amd_ryzen_plugin import create_amd_ryzen_plugin

logger = logging.getLogger(__name__)


def get_processor_plugin(
    profile: Optional[SystemProfile] = None,
) -> HardwareProcessorPluginInterface:
    """
    Automatically selects and instantiates the best processor plugin.

    Args:
        profile: Optional SystemProfile. If not provided, it will be analyzed
                 automatically.

    Returns:
        An instance of a class implementing HardwareProcessorPluginInterface.
    """
    if profile is None:
        profile = get_system_profile()

    cpu_brand = profile.cpu_brand
    logger.info(f"Auto-detecting processor plugin for CPU: {cpu_brand}")

    # Normalize for case-insensitive matching
    cpu_brand_lower = cpu_brand.lower()

    # Intel Kaby Lake (i5-7500)
    is_kaby_lake = (
        "i5-7500" in cpu_brand or
        ("kaby lake" in cpu_brand_lower and "i5" in cpu_brand_lower)
    )
    if is_kaby_lake:
        logger.info(
            "Detected Intel Kaby Lake (i5-7500). Using IntelKabyLakePlugin."
        )
        return create_intel_kaby_lake_plugin(profile)

    # AMD Ryzen (Ryzen 7 5700)
    # Matching "Ryzen 7 5700" or generic strong Ryzen
    if "ryzen 7 5700" in cpu_brand_lower:
        logger.info("Detected AMD Ryzen 7 5700. Using AmdRyzenPlugin.")
        return create_amd_ryzen_plugin(profile)

    # Generic Ryzen fallback (for 5000 series or similar)
    if "ryzen" in cpu_brand_lower and "amd" in cpu_brand_lower:
        logger.info(
            f"Detected Generic AMD Ryzen ({cpu_brand}). "
            f"Using AmdRyzenPlugin."
        )
        return create_amd_ryzen_plugin(profile)

    # Intel Comet Lake (i5-10210U)
    if "i5-10210u" in cpu_brand_lower or "comet lake" in cpu_brand_lower:
        logger.info(
            "Detected Intel Comet Lake (i5-10210U). "
            "Using IntelCometLakePlugin."
        )
        return create_intel_comet_lake_plugin(profile)

    # Generic Fallback
    logger.info(
        f"No specific plugin for {cpu_brand}. Using GenericCPUPlugin."
    )
    return create_generic_cpu_plugin()
