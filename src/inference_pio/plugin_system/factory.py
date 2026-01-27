"""
Processor Plugin Factory

This module provides a factory to create the most appropriate processor plugin
based on the system hardware profile.
"""

import logging
from ..common.hardware_analyzer import SystemProfile
from .processor_interface import ProcessorPluginInterface
from .cpu_plugin import create_generic_cpu_plugin
from .intel_comet_lake_plugin import create_intel_comet_lake_plugin

logger = logging.getLogger(__name__)

def get_processor_plugin(profile: SystemProfile) -> ProcessorPluginInterface:
    """
    Factory function to get the best processor plugin for the current hardware.

    Args:
        profile: The system hardware profile.

    Returns:
        An instantiated ProcessorPluginInterface.
    """
    cpu_brand = profile.cpu_brand.lower()

    # Check for Intel Comet Lake i5-10210U
    if "i5-10210u" in cpu_brand or ("10210u" in cpu_brand and "intel" in cpu_brand):
        logger.info(f"Detected Intel Comet Lake CPU ({profile.cpu_brand}). Using specialized plugin.")
        return create_intel_comet_lake_plugin(profile)

    # Check for other Intel CPUs (Future expansion)
    # if "intel" in cpu_brand: ...

    # Default to Generic CPU
    logger.info(f"Using Generic CPU Plugin for: {profile.cpu_brand}")
    return create_generic_cpu_plugin()
