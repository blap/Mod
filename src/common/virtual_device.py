"""
Virtual Device Simulation System

This module implements a virtual device simulation system that allows distributing
model execution across multiple simulated devices even on a single physical device.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VirtualDeviceState(Enum):
    """State of a virtual device."""

    IDLE = "idle"
    COMPUTING = "computing"
    MEMORY_TRANSFER = "memory_transfer"
    SYNCHRONIZING = "synchronizing"


@dataclass
class VirtualDevice:
    """Represents a virtual device (e.g. GPU)."""

    id: int
    memory_limit_gb: float
    compute_capability: str = "7.5"  # Default to V100-like capability
    name: str = "Virtual_Device"
    state: VirtualDeviceState = VirtualDeviceState.IDLE
    current_memory_usage_gb: float = 0.0
    peak_memory_usage_gb: float = 0.0
    allocated_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    active: bool = True
    utilization: float = 0.0  # Current utilization percentage


class VirtualDeviceSimulator:
    """
    Simulates multiple devices on a single physical device.
    """

    def __init__(self, num_virtual_devices: int = 2, memory_per_device_gb: float = 4.0):
        self.num_virtual_devices = num_virtual_devices
        self.memory_per_device_gb = memory_per_device_gb
        self.virtual_devices: List[VirtualDevice] = []
        self.device_locks = [threading.Lock() for _ in range(num_virtual_devices)]
        self.global_lock = threading.Lock()

        # Initialize virtual devices
        self._initialize_virtual_devices()

        logger.info(
            f"Virtual device simulator initialized with {num_virtual_devices} "
            f"virtual devices, each with {memory_per_device_gb}GB memory"
        )

    def _initialize_virtual_devices(self):
        """Initialize virtual devices."""
        for i in range(self.num_virtual_devices):
            device = VirtualDevice(
                id=i,
                memory_limit_gb=self.memory_per_device_gb,
                name=f"Virtual_Device_{i}",
                state=VirtualDeviceState.IDLE,
            )
            self.virtual_devices.append(device)

    def allocate_on_device(
        self, device_id: int, tensor_name: str, tensor: torch.Tensor
    ) -> bool:
        """
        Allocate a tensor on a specific virtual device.

        Args:
            device_id: ID of the virtual device
            tensor_name: Name to assign to the tensor
            tensor: The tensor to allocate

        Returns:
            True if allocation was successful, False otherwise
        """
        if device_id >= len(self.virtual_devices):
            logger.error(f"Invalid device ID: {device_id}")
            return False

        with self.device_locks[device_id]:
            device = self.virtual_devices[device_id]

            # Calculate memory requirement
            memory_required_gb = self._calculate_tensor_memory(tensor)

            # Check if there's enough memory available
            available_memory = device.memory_limit_gb - device.current_memory_usage_gb
            if memory_required_gb > available_memory:
                logger.warning(
                    f"Not enough memory on virtual device {device_id}. "
                    f"Required: {memory_required_gb:.2f}GB, "
                    f"Available: {available_memory:.2f}GB"
                )
                return False

            # Update device state
            device.state = VirtualDeviceState.COMPUTING
            device.current_memory_usage_gb += memory_required_gb
            device.utilization = device.current_memory_usage_gb / device.memory_limit_gb

            if device.current_memory_usage_gb > device.peak_memory_usage_gb:
                device.peak_memory_usage_gb = device.current_memory_usage_gb

            # Store tensor reference (in simulation, we just track it)
            device.allocated_tensors[tensor_name] = tensor

            logger.debug(
                f"Allocated tensor '{tensor_name}' ({memory_required_gb:.2f}GB) "
                f"on virtual device {device_id}"
            )
            return True

    def deallocate_from_device(self, device_id: int, tensor_name: str) -> bool:
        """
        Deallocate a tensor from a specific virtual device.

        Args:
            device_id: ID of the virtual device
            tensor_name: Name of the tensor to deallocate

        Returns:
            True if deallocation was successful, False otherwise
        """
        if device_id >= len(self.virtual_devices):
            logger.error(f"Invalid device ID: {device_id}")
            return False

        with self.device_locks[device_id]:
            device = self.virtual_devices[device_id]

            if tensor_name not in device.allocated_tensors:
                logger.warning(
                    f"Tensor '{tensor_name}' not found on virtual device {device_id}"
                )
                return False

            tensor = device.allocated_tensors[tensor_name]
            memory_released_gb = self._calculate_tensor_memory(tensor)

            # Update device state
            device.current_memory_usage_gb -= memory_released_gb
            device.utilization = max(
                0.0, device.current_memory_usage_gb / device.memory_limit_gb
            )

            # Remove tensor reference
            del device.allocated_tensors[tensor_name]

            # Update state if no tensors remain
            if len(device.allocated_tensors) == 0:
                device.state = VirtualDeviceState.IDLE

            logger.debug(
                f"Deallocated tensor '{tensor_name}' ({memory_released_gb:.2f}GB) "
                f"from virtual device {device_id}"
            )
            return True

    def _calculate_tensor_memory(self, tensor: torch.Tensor) -> float:
        """Calculate memory usage of a tensor in GB."""
        element_size_bytes = tensor.element_size()
        num_elements = tensor.nelement()
        memory_bytes = element_size_bytes * num_elements
        return memory_bytes / (1024**3)  # Convert to GB

    def simulate_compute(
        self, device_id: int, operation: str, duration_ms: float = 1.0
    ) -> bool:
        """
        Simulate a compute operation on a virtual device.

        Args:
            device_id: ID of the virtual device
            operation: Description of the operation
            duration_ms: Duration of the operation in milliseconds

        Returns:
            True if operation was successful, False otherwise
        """
        if device_id >= len(self.virtual_devices):
            logger.error(f"Invalid device ID: {device_id}")
            return False

        with self.device_locks[device_id]:
            device = self.virtual_devices[device_id]

            # Update state
            prev_state = device.state
            device.state = VirtualDeviceState.COMPUTING

            logger.debug(
                f"Starting operation '{operation}' on virtual device {device_id} "
                f"for {duration_ms}ms"
            )

            # Simulate computation time
            time.sleep(duration_ms / 1000.0)  # Convert ms to seconds

            # Update state back
            device.state = prev_state

            logger.debug(
                f"Completed operation '{operation}' on virtual device {device_id}"
            )
            return True

    def simulate_memory_transfer(
        self,
        source_device_id: int,
        dest_device_id: int,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> bool:
        """
        Simulate memory transfer between virtual devices.

        Args:
            source_device_id: Source virtual device ID
            dest_device_id: Destination virtual device ID
            tensor_name: Name of the tensor to transfer
            tensor: The tensor to transfer

        Returns:
            True if transfer was successful, False otherwise
        """
        if source_device_id >= len(self.virtual_devices) or dest_device_id >= len(
            self.virtual_devices
        ):
            logger.error(
                f"Invalid device ID: source={source_device_id}, dest={dest_device_id}"
            )
            return False

        # Calculate transfer time based on tensor size
        memory_gb = self._calculate_tensor_memory(tensor)
        transfer_time_ms = memory_gb * 100  # Simulated transfer rate: 10GB/s

        with self.global_lock:
            # Update source device state
            with self.device_locks[source_device_id]:
                self.virtual_devices[source_device_id].state = (
                    VirtualDeviceState.MEMORY_TRANSFER
                )

            # Update destination device state
            with self.device_locks[dest_device_id]:
                self.virtual_devices[dest_device_id].state = (
                    VirtualDeviceState.MEMORY_TRANSFER
                )

            logger.debug(
                f"Transferring tensor '{tensor_name}' ({memory_gb:.2f}GB) "
                f"from virtual device {source_device_id} to {dest_device_id} "
                f"(estimated time: {transfer_time_ms:.2f}ms)"
            )

            # Simulate transfer time
            time.sleep(transfer_time_ms / 1000.0)

            # Update states back to previous state
            with self.device_locks[source_device_id]:
                if len(self.virtual_devices[source_device_id].allocated_tensors) == 0:
                    self.virtual_devices[source_device_id].state = (
                        VirtualDeviceState.IDLE
                    )
                else:
                    self.virtual_devices[source_device_id].state = (
                        VirtualDeviceState.COMPUTING
                    )

            with self.device_locks[dest_device_id]:
                if len(self.virtual_devices[dest_device_id].allocated_tensors) == 0:
                    self.virtual_devices[dest_device_id].state = VirtualDeviceState.IDLE
                else:
                    self.virtual_devices[dest_device_id].state = (
                        VirtualDeviceState.COMPUTING
                    )

        logger.debug(
            f"Completed transfer of tensor '{tensor_name}' from virtual device "
            f"{source_device_id} to {dest_device_id}"
        )
        return True

    def synchronize_devices(self, device_ids: List[int] = None) -> bool:
        """
        Synchronize virtual devices.

        Args:
            device_ids: List of device IDs to synchronize. If None, synchronize all devices.

        Returns:
            True if synchronization was successful, False otherwise
        """
        if device_ids is None:
            device_ids = list(range(len(self.virtual_devices)))

        # Validate device IDs
        for device_id in device_ids:
            if device_id >= len(self.virtual_devices):
                logger.error(f"Invalid device ID: {device_id}")
                return False

        with self.global_lock:
            # Set all specified devices to synchronizing state
            for device_id in device_ids:
                with self.device_locks[device_id]:
                    self.virtual_devices[device_id].state = (
                        VirtualDeviceState.SYNCHRONIZING
                    )

            logger.debug(f"Synchronizing virtual devices: {device_ids}")

            # Simulate synchronization time
            time.sleep(0.001)  # 1ms sync time

            # Reset states to idle or computing based on tensor allocation
            for device_id in device_ids:
                with self.device_locks[device_id]:
                    if len(self.virtual_devices[device_id].allocated_tensors) == 0:
                        self.virtual_devices[device_id].state = VirtualDeviceState.IDLE
                    else:
                        self.virtual_devices[device_id].state = (
                            VirtualDeviceState.COMPUTING
                        )

        logger.debug(f"Completed synchronization of virtual devices: {device_ids}")
        return True

    def get_device_stats(
        self, device_id: int = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get statistics for virtual devices.

        Args:
            device_id: Specific device ID to get stats for. If None, get stats for all devices.

        Returns:
            Device statistics
        """
        if device_id is not None:
            if device_id >= len(self.virtual_devices):
                logger.error(f"Invalid device ID: {device_id}")
                return {}

            device = self.virtual_devices[device_id]
            return {
                "id": device.id,
                "name": device.name,
                "state": device.state.value,
                "memory_limit_gb": device.memory_limit_gb,
                "current_memory_usage_gb": device.current_memory_usage_gb,
                "peak_memory_usage_gb": device.peak_memory_usage_gb,
                "utilization": device.utilization,
                "active": device.active,
                "num_allocated_tensors": len(device.allocated_tensors),
            }
        else:
            stats_list = []
            for device in self.virtual_devices:
                stats = {
                    "id": device.id,
                    "name": device.name,
                    "state": device.state.value,
                    "memory_limit_gb": device.memory_limit_gb,
                    "current_memory_usage_gb": device.current_memory_usage_gb,
                    "peak_memory_usage_gb": device.peak_memory_usage_gb,
                    "utilization": device.utilization,
                    "active": device.active,
                    "num_allocated_tensors": len(device.allocated_tensors),
                }
                stats_list.append(stats)
            return stats_list

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics for the virtual device simulator."""
        total_memory = sum(device.memory_limit_gb for device in self.virtual_devices)
        used_memory = sum(
            device.current_memory_usage_gb for device in self.virtual_devices
        )
        peak_memory = sum(
            device.peak_memory_usage_gb for device in self.virtual_devices
        )

        active_devices = sum(1 for device in self.virtual_devices if device.active)
        busy_devices = sum(
            1
            for device in self.virtual_devices
            if device.state != VirtualDeviceState.IDLE
        )

        return {
            "num_virtual_devices": len(self.virtual_devices),
            "num_active_devices": active_devices,
            "num_busy_devices": busy_devices,
            "total_memory_gb": total_memory,
            "used_memory_gb": used_memory,
            "peak_memory_gb": peak_memory,
            "memory_utilization_ratio": (
                used_memory / total_memory if total_memory > 0 else 0
            ),
            "average_utilization": (
                sum(device.utilization for device in self.virtual_devices)
                / len(self.virtual_devices)
                if self.virtual_devices
                else 0
            ),
        }

    def cleanup(self):
        """Clean up resources used by the virtual device simulator."""
        # Clear all allocated tensors
        for device in self.virtual_devices:
            device.allocated_tensors.clear()
            device.current_memory_usage_gb = 0.0
            device.peak_memory_usage_gb = 0.0
            device.state = VirtualDeviceState.IDLE
            device.utilization = 0.0

        logger.info("Virtual device simulator resources cleaned up")

    def get_memory_pressure(self, device_id: int = None) -> float:
        """
        Get memory pressure for a specific virtual device or overall.

        Args:
            device_id: ID of virtual device to check. If None, return average pressure.

        Returns:
            Memory pressure ratio (0.0 to 1.0)
        """
        if device_id is not None:
            if device_id < len(self.virtual_devices):
                device = self.virtual_devices[device_id]
                return (
                    device.current_memory_usage_gb / device.memory_limit_gb
                    if device.memory_limit_gb > 0
                    else 0.0
                )
            else:
                return 0.0
        else:
            # Return average memory pressure across all virtual devices
            if not self.virtual_devices:
                return 0.0

            total_pressure = 0.0
            for device in self.virtual_devices:
                pressure = (
                    device.current_memory_usage_gb / device.memory_limit_gb
                    if device.memory_limit_gb > 0
                    else 0.0
                )
                total_pressure += pressure

            return total_pressure / len(self.virtual_devices)

    def trigger_memory_management(self, threshold: float = 0.8):
        """
        Trigger memory management when memory pressure exceeds threshold.

        Args:
            threshold: Memory pressure threshold (0.0 to 1.0) that triggers management
        """
        for i, device in enumerate(self.virtual_devices):
            pressure = self.get_memory_pressure(i)
            if pressure > threshold:
                logger.info(
                    f"High memory pressure ({pressure:.2f}) on virtual device {i}, triggering management"
                )

                # In a real implementation, we would take action to reduce memory usage
                # For simulation, we'll just log the event
                self._handle_high_memory_pressure(i, pressure)

    def _handle_high_memory_pressure(self, device_id: int, pressure: float):
        """Handle high memory pressure on a specific device."""
        # In a real implementation, this would evict tensors or take other actions
        logger.debug(
            f"Handling high memory pressure ({pressure:.2f}) on virtual device {device_id}"
        )


class VirtualExecutionSimulator:
    """
    Main class that combines virtual device simulation with distributed execution.
    """

    def __init__(self, num_virtual_devices: int = 2, memory_per_device_gb: float = 4.0):
        self.virtual_device_simulator = VirtualDeviceSimulator(
            num_virtual_devices=num_virtual_devices,
            memory_per_device_gb=memory_per_device_gb,
        )
        self.execution_lock = threading.Lock()

    def execute_partition_on_device(
        self,
        partition: nn.Module,
        input_tensor: torch.Tensor,
        device_id: int,
        partition_name: str = "partition",
    ) -> torch.Tensor:
        """
        Execute a model partition on a specific virtual device.

        Args:
            partition: The model partition to execute
            input_tensor: Input tensor for the partition
            device_id: Virtual device ID to execute on
            partition_name: Name of the partition for logging

        Returns:
            Output tensor from the partition
        """
        with self.execution_lock:
            # Monitor memory pressure before execution
            memory_pressure = self.virtual_device_simulator.get_memory_pressure(
                device_id
            )
            if memory_pressure > 0.7:  # Trigger management if memory usage is above 70%
                logger.info(
                    f"High memory pressure ({memory_pressure:.2f}) on virtual device {device_id} before execution, triggering management"
                )
                self.virtual_device_simulator.trigger_memory_management(0.7)

            # Allocate input tensor on the virtual device
            input_name = f"{partition_name}_input"
            if not self.virtual_device_simulator.allocate_on_device(
                device_id, input_name, input_tensor
            ):
                logger.error(
                    f"Failed to allocate input tensor on virtual device {device_id}"
                )
                raise RuntimeError(
                    f"Failed to allocate input tensor on virtual device {device_id}"
                )

            # Allocate partition parameters on the virtual device
            for name, param in partition.named_parameters():
                param_name = f"{partition_name}_{name}"
                if not self.virtual_device_simulator.allocate_on_device(
                    device_id, param_name, param
                ):
                    logger.warning(
                        f"Failed to allocate parameter '{param_name}' on virtual device {device_id}"
                    )
                    # Continue execution even if parameter allocation fails (simulation)

            # Simulate computation on the virtual device
            device_compute_time = self._estimate_compute_time(partition, input_tensor)
            self.virtual_device_simulator.simulate_compute(
                device_id, f"execute_{partition_name}", duration_ms=device_compute_time
            )

            # Execute the partition (in simulation, we just pass through)
            device = torch.device(
                f"cuda:{device_id % torch.cuda.device_count()}"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
            partition = partition.to(device)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output_tensor = partition(input_tensor)

            # Deallocate input tensor (parameters stay allocated for subsequent runs)
            self.virtual_device_simulator.deallocate_from_device(device_id, input_name)

            # Allocate output tensor on the virtual device
            output_name = f"{partition_name}_output"
            if not self.virtual_device_simulator.allocate_on_device(
                device_id, output_name, output_tensor
            ):
                logger.warning(
                    f"Failed to allocate output tensor on virtual device {device_id}"
                )

            # Monitor memory pressure after execution
            memory_pressure = self.virtual_device_simulator.get_memory_pressure(
                device_id
            )
            if memory_pressure > 0.8:  # Trigger management if memory usage is above 80%
                logger.info(
                    f"High memory pressure ({memory_pressure:.2f}) on virtual device {device_id} after execution, triggering management"
                )
                self.virtual_device_simulator.trigger_memory_management(0.8)

            logger.debug(
                f"Executed partition '{partition_name}' on virtual device {device_id}"
            )
            return output_tensor

    def _estimate_compute_time(
        self, partition: nn.Module, input_tensor: torch.Tensor
    ) -> float:
        """
        Estimate compute time for a partition based on its complexity.

        Args:
            partition: The model partition
            input_tensor: Input tensor to the partition

        Returns:
            Estimated compute time in milliseconds
        """
        # Simple heuristic: more parameters and larger input = longer compute time
        num_params = sum(p.numel() for p in partition.parameters())
        input_size = input_tensor.nelement()

        # Base time plus scaling factors
        base_time_ms = 0.1
        param_factor = num_params / 1e6  # Scale by millions of parameters
        input_factor = input_size / 1e4  # Scale by ten thousands of elements

        estimated_time = base_time_ms + (param_factor * 0.5) + (input_factor * 0.1)

        return min(estimated_time, 100.0)  # Cap at 100ms for simulation

    def transfer_tensor_between_devices(
        self,
        tensor: torch.Tensor,
        tensor_name: str,
        source_device_id: int,
        dest_device_id: int,
    ) -> bool:
        """
        Transfer a tensor between virtual devices.

        Args:
            tensor: The tensor to transfer
            tensor_name: Name of the tensor
            source_device_id: Source virtual device ID
            dest_device_id: Destination virtual device ID

        Returns:
            True if transfer was successful, False otherwise
        """
        success = self.virtual_device_simulator.simulate_memory_transfer(
            source_device_id, dest_device_id, tensor_name, tensor
        )

        if success:
            # Update allocations
            self.virtual_device_simulator.deallocate_from_device(
                source_device_id, tensor_name
            )
            self.virtual_device_simulator.allocate_on_device(
                dest_device_id, tensor_name, tensor
            )

        return success

    def synchronize_all_devices(self) -> bool:
        """Synchronize all virtual devices."""
        device_ids = list(range(self.virtual_device_simulator.num_virtual_devices))
        return self.virtual_device_simulator.synchronize_devices(device_ids)

    def synchronize_subset_of_devices(self, device_ids: List[int]) -> bool:
        """
        Synchronize a subset of virtual devices.

        Args:
            device_ids: List of device IDs to synchronize

        Returns:
            True if synchronization was successful, False otherwise
        """
        return self.virtual_device_simulator.synchronize_devices(device_ids)

    def pipeline_synchronize(self, current_stage: int, num_stages: int) -> bool:
        """
        Synchronize devices in a pipeline fashion.

        Args:
            current_stage: Current pipeline stage
            num_stages: Total number of pipeline stages

        Returns:
            True if synchronization was successful, False otherwise
        """
        # In a pipeline, we might only need to synchronize adjacent stages
        # For simulation, we'll just do a general sync
        logger.debug(f"Pipeline synchronization for stage {current_stage}/{num_stages}")

        # Determine which devices to synchronize based on pipeline stage
        device_ids = [current_stage % self.virtual_device_simulator.num_virtual_devices]

        # If we have multiple devices per stage, add them too
        devices_per_stage = max(
            1, self.virtual_device_simulator.num_virtual_devices // num_stages
        )
        for i in range(devices_per_stage):
            device_id = (
                current_stage * devices_per_stage + i
            ) % self.virtual_device_simulator.num_virtual_devices
            if device_id not in device_ids:
                device_ids.append(device_id)

        return self.virtual_device_simulator.synchronize_devices(device_ids)

    def event_based_sync(self, event_name: str, timeout: float = 1.0) -> bool:
        """
        Perform event-based synchronization.

        Args:
            event_name: Name of the event to wait for
            timeout: Timeout in seconds

        Returns:
            True if synchronization was successful, False otherwise
        """
        logger.debug(
            f"Event-based synchronization for event '{event_name}' with timeout {timeout}s"
        )

        # In a real implementation, this would wait for a specific event
        # For simulation, we'll just sleep briefly
        time.sleep(0.001)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics for the virtual execution simulator."""
        return {
            "virtual_device_stats": self.virtual_device_simulator.get_overall_stats(),
            "device_details": self.virtual_device_simulator.get_device_stats(),
        }

    def cleanup(self):
        """Clean up resources."""
        self.virtual_device_simulator.cleanup()

    def monitor_memory_pressure(self, threshold: float = 0.8) -> bool:
        """
        Monitor memory pressure and trigger management if needed.

        Args:
            threshold: Memory pressure threshold (0.0 to 1.0) that triggers management

        Returns:
            True if memory management was triggered, False otherwise
        """
        pressure = self.virtual_device_simulator.get_memory_pressure()
        if pressure > threshold:
            logger.info(
                f"High overall memory pressure ({pressure:.2f}), triggering memory management"
            )
            self.virtual_device_simulator.trigger_memory_management(threshold)
            return True
        return False
