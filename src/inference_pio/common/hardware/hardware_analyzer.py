"""
Hardware Analyzer for Inference-PIO

This module provides tools to analyze the system hardware capabilities,
identifying CPU cores, RAM, VRAM, and Disk space to optimize model execution.
It is specifically designed to detect "weak" hardware and suggest appropriate
optimizations like aggressive offloading or specific processor plugins.
"""

import logging
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class SystemProfile:
    """Data class representing the system hardware profile."""

    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    available_vram_gb: float
    gpu_name: str
    disk_free_gb: float
    is_weak_hardware: bool
    recommended_offload_strategy: str
    safe_vram_limit_gb: float
    processor_architecture: str
    cpu_brand: str
    instruction_sets: List[str]
    secondary_gpu_detected: str = "None"
    hybrid_capability: bool = False
    ipex_available: bool = False

    def get_cpu_allocation(self) -> int:
        """
        Return recommended number of CPU threads to use.
        Leaves one core free for system if possible.
        """
        if self.cpu_cores_physical > 1:
            return self.cpu_cores_physical  # Use all physical cores for heavy compute
        return 1


class HardwareAnalyzer:
    """
    Analyzes system hardware to determine capabilities and constraints.
    """

    def __init__(
        self, weak_ram_threshold_gb: float = 16.0, weak_vram_threshold_gb: float = 8.0
    ):
        self.weak_ram_threshold_gb = weak_ram_threshold_gb
        self.weak_vram_threshold_gb = weak_vram_threshold_gb
        self._profile = None

    def analyze(self) -> SystemProfile:
        """
        Perform a full analysis of the system hardware.
        """
        if self._profile:
            return self._profile

        # CPU Analysis
        cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        processor_architecture = platform.machine()

        # Get CPU Brand
        cpu_brand = platform.processor()
        # Fallback/refinement for Linux
        if platform.system() == "Linux":
            try:
                command = "cat /proc/cpuinfo | grep 'model name' | head -1"
                output = subprocess.check_output(command, shell=True).decode().strip()
                if ":" in output:
                    cpu_brand = output.split(":", 1)[1].strip()
            except Exception:
                # Placeholder for actual hardware analysis implementation
                # This would contain the actual hardware analysis algorithm
                logger.warning("Hardware analysis not implemented for this operation")
                pass
        # Fallback for Windows (often platform.processor() is generic on some python versions, but usually ok)

        # RAM Analysis
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024**3)
        available_ram_gb = vm.available / (1024**3)

        # GPU/VRAM Analysis
        total_vram_gb = 0.0
        available_vram_gb = 0.0
        gpu_name = "None"
        secondary_gpu_detected = "None"
        hybrid_capability = False
        ipex_available = False

        # Check for Intel Extension for PyTorch (IPEX)
        try:
            import intel_extension_for_pytorch as ipex

            ipex_available = True
            # Try to detect XPU devices if available
            try:
                if hasattr(torch, "xpu") and torch.xpu.device_count() > 0:
                    secondary_gpu_detected = torch.xpu.get_device_name(0)
            except Exception:
                # Placeholder for actual hardware analysis implementation
                # This would contain the actual hardware analysis algorithm
                logger.warning("Hardware analysis not implemented for this operation")
                pass
        except ImportError:
            # If detection fails, return a default value
            return 0

        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    # Use the first GPU for primary stats or sum them up?
                    # For simplicity, let's look at device 0 or sum total if using multiple
                    # Here we focus on total available VRAM across all devices for "giant model" potential
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        total_vram_gb += props.total_memory / (1024**3)
                        # Available is harder to get directly from torch without pynvml,
                        # but we can approximate or assume close to total if we are the main process
                        # torch.cuda.mem_get_info() returns (free, total)
                        free_mem, _ = torch.cuda.mem_get_info(i)
                        available_vram_gb += free_mem / (1024**3)

                    gpu_name = torch.cuda.get_device_name(0)
            except Exception as e:
                logger.warning(f"Error detecting CUDA VRAM: {e}")

        # Detect Hybrid Scenario (e.g. NVIDIA + Intel Integrated/Arc or CUDA + CPU offload needed)
        # If we have a dGPU but it's small, and a decent CPU, we are in a hybrid scenario.
        # Also if we explicitly detected a secondary XPU.
        if gpu_name != "None" and (
            ipex_available or secondary_gpu_detected != "None" or total_vram_gb < 4.0
        ):
            hybrid_capability = True

        # Disk Analysis
        # Check current working directory disk usage
        try:
            disk_usage = psutil.disk_usage(".")
            disk_free_gb = disk_usage.free / (1024**3)
        except Exception as e:
            logger.warning(f"Error detecting disk space: {e}")
            disk_free_gb = 0.0

        # Instruction Sets (Basic detection)
        instruction_sets = self._detect_instruction_sets()

        # Weak Hardware Classification
        # We consider it weak if RAM or VRAM is below thresholds, necessitating offloading
        is_weak_hardware = (total_ram_gb < self.weak_ram_threshold_gb) or (
            total_vram_gb < self.weak_vram_threshold_gb
        )

        # Recommendation
        if is_weak_hardware:
            # Lower threshold for hybrid aggressive to utilize any available VRAM (e.g. MX330 2GB)
            if available_vram_gb > 0.5:
                recommended_offload_strategy = (
                    "hybrid_aggressive"  # Use VRAM for critical paths, offload rest
                )
            else:
                recommended_offload_strategy = "disk_only"  # Mostly CPU/Disk
        else:
            recommended_offload_strategy = "none"

        # Safe VRAM Limit (Leave some buffer for system)
        # If we have VRAM, safe limit is e.g. 90% of available
        safe_vram_limit_gb = max(0.0, available_vram_gb * 0.9)

        self._profile = SystemProfile(
            cpu_cores_physical=cpu_cores_physical,
            cpu_cores_logical=cpu_cores_logical,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            total_vram_gb=total_vram_gb,
            available_vram_gb=available_vram_gb,
            gpu_name=gpu_name,
            disk_free_gb=disk_free_gb,
            is_weak_hardware=is_weak_hardware,
            recommended_offload_strategy=recommended_offload_strategy,
            safe_vram_limit_gb=safe_vram_limit_gb,
            processor_architecture=processor_architecture,
            cpu_brand=cpu_brand,
            instruction_sets=instruction_sets,
            secondary_gpu_detected=secondary_gpu_detected,
            hybrid_capability=hybrid_capability,
            ipex_available=ipex_available,
        )

        return self._profile

    def _detect_instruction_sets(self) -> List[str]:
        """
        Attempt to detect supported instruction sets (AVX, AVX2, NEON, etc).
        """
        sets = []
        try:
            # Linux / macOS
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    info = f.read()
                    if "avx2" in info:
                        sets.append("AVX2")
                    elif "avx" in info:
                        sets.append("AVX")
                    if "fma" in info:
                        sets.append("FMA")
                    if "sse4_2" in info:
                        sets.append("SSE4.2")
            elif platform.system() == "Darwin":
                # Apple Silicon often implies NEON/ARM features
                # sysctl -a | grep machdep.cpu.features
                # This is a basic check
                machine = platform.machine()
                if "arm" in machine.lower() or "aarch64" in machine.lower():
                    sets.append("NEON")
                    sets.append("ARMv8")
            elif platform.system() == "Windows":
                # Windows detection is tricker without external libs or wmic
                # We can infer from env or just leave empty for generic fallback
                # Placeholder for actual hardware analysis implementation
                # This would contain the actual hardware analysis algorithm
                logger.warning("Hardware analysis not implemented for this operation")
                pass
        except Exception:
            # If detection fails, return an empty list
            return []

        return sets

    def get_cpu_allocation(self) -> int:
        """
        Return recommended number of CPU threads to use.
        Leaves one core free for system if possible.
        """
        profile = self.analyze()
        if profile.cpu_cores_physical > 1:
            return (
                profile.cpu_cores_physical
            )  # Use all physical cores for heavy compute
        return 1


# Global instance
_analyzer = HardwareAnalyzer()


def get_system_profile() -> SystemProfile:
    return _analyzer.analyze()
