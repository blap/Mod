"""
System Check Tool

This module provides functionality to check system hardware and compatibility
for running inference models.
"""

import os
import platform
import psutil
import logging
from typing import Dict, Any, List

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        brand = info.get('brand_raw', 'Unknown')
    except ImportError:
        brand = platform.processor()

    return {
        "brand": brand,
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "usage_percent": psutil.cpu_percent(interval=0.1),
        "frequency_current": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
    }

def get_memory_info() -> Dict[str, Any]:
    """Get system memory information."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        "total_gb": vm.total / (1024**3),
        "available_gb": vm.available / (1024**3),
        "used_gb": vm.used / (1024**3),
        "percent": vm.percent,
        "swap_total_gb": swap.total / (1024**3),
        "swap_used_gb": swap.used / (1024**3),
    }

def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information if available."""
    gpus = []

    # Check NVIDIA GPUs via PyTorch/pynvml
    if torch and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)

            gpus.append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "allocated_memory_gb": mem_allocated,
                "reserved_memory_gb": mem_reserved,
                "type": "NVIDIA (CUDA)",
            })

    # Fallback/Additional check using pynvml if needed could go here
    # For now, torch.cuda is the primary way we interact with GPUs

    return gpus

def get_disk_info() -> Dict[str, Any]:
    """Get disk information, specifically checking H: drive if on Windows."""
    disks = []

    partitions = psutil.disk_partitions()
    h_drive_found = False

    for p in partitions:
        try:
            usage = psutil.disk_usage(p.mountpoint)
            disk_info = {
                "device": p.device,
                "mountpoint": p.mountpoint,
                "fstype": p.fstype,
                "total_gb": usage.total / (1024**3),
                "free_gb": usage.free / (1024**3),
            }
            disks.append(disk_info)

            # Check for H: drive specifically
            if platform.system() == 'Windows' and 'H:' in p.device:
                h_drive_found = True
            elif platform.system() != 'Windows' and '/mnt/h' in p.mountpoint: # WSL/Linux assumption
                h_drive_found = True

        except PermissionError:
            continue

    return {
        "disks": disks,
        "h_drive_detected": h_drive_found
    }

def perform_system_check() -> Dict[str, Any]:
    """Perform a full system check."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpus": get_gpu_info(),
        "disks": get_disk_info(),
        "torch_version": torch.__version__ if torch else "Not installed",
        "cuda_available": torch.cuda.is_available() if torch else False,
    }

def print_system_check_report(report: Dict[str, Any]):
    """Print the system check report to stdout using Rich."""
    from .rich_utils import console
    from rich.table import Table
    from rich.panel import Panel

    # System Info Table
    sys_table = Table(title="System Information", show_header=False)
    sys_table.add_column("Property", style="cyan")
    sys_table.add_column("Value", style="white")
    sys_table.add_row("OS", report['os'])
    sys_table.add_row("Python", report['python_version'])
    sys_table.add_row("PyTorch", report['torch_version'])

    cuda_style = "bold green" if report['cuda_available'] else "bold red"
    sys_table.add_row("CUDA Available", str(report['cuda_available']), style=cuda_style)

    console.print(sys_table)

    # CPU Info
    cpu_table = Table(title="CPU", show_header=False)
    cpu_table.add_column("Property", style="cyan")
    cpu_table.add_column("Value", style="white")
    cpu_table.add_row("Brand", report['cpu']['brand'])
    cpu_table.add_row("Cores", f"{report['cpu']['cores_physical']} physical, {report['cpu']['cores_logical']} logical")
    cpu_table.add_row("Usage", f"{report['cpu']['usage_percent']}%")
    console.print(cpu_table)

    # Memory Info
    mem_table = Table(title="Memory", show_header=False)
    mem_table.add_column("Property", style="cyan")
    mem_table.add_column("Value", style="white")
    mem_table.add_row("Total", f"{report['memory']['total_gb']:.2f} GB")
    mem_table.add_row("Available", f"{report['memory']['available_gb']:.2f} GB")
    mem_table.add_row("Used", f"{report['memory']['percent']}%")
    console.print(mem_table)

    # GPU Info
    if report['gpus']:
        gpu_table = Table(title="GPUs")
        gpu_table.add_column("ID", style="dim")
        gpu_table.add_column("Name", style="bold")
        gpu_table.add_column("Total Memory", style="green")
        gpu_table.add_column("Allocated", style="yellow")

        for gpu in report['gpus']:
            gpu_table.add_row(
                str(gpu['id']),
                gpu['name'],
                f"{gpu['total_memory_gb']:.2f} GB",
                f"{gpu['allocated_memory_gb']:.2f} GB"
            )
        console.print(gpu_table)
    else:
        console.print(Panel("No GPUs detected", title="GPUs", border_style="red"))

    # Storage Info
    storage_table = Table(title="Storage Check", show_header=False)
    storage_table.add_column("Property", style="cyan")
    storage_table.add_column("Value", style="white")

    h_style = "bold green" if report['disks']['h_drive_detected'] else "bold yellow"
    storage_table.add_row("H: Drive Detected", str(report['disks']['h_drive_detected']), style=h_style)
    console.print(storage_table)
