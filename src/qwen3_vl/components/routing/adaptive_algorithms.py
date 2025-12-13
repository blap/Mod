"""
Adaptive Algorithms for Power and Thermal Constraints

This module implements adaptive algorithms that adjust model behavior based on
power and thermal constraints to maintain performance while staying within limits.

The module provides:
- Adaptive parameter management based on system constraints
- Load balancing strategies for distributed workloads
- Adaptive model wrappers for machine learning applications
- Historical trend analysis for optimization

The adaptive algorithms are designed to work with power and thermal management
systems to dynamically adjust model execution parameters based on real-time
system conditions. This helps maintain performance while preventing overheating
and excessive power consumption.

Classes:
    AdaptiveParameters: Container for adaptive parameters
    AdaptationStrategy: Enum for different adaptation strategies
    AdaptiveController: Main controller for adaptive parameter management
    LoadBalancer: Adaptive load balancer for distributed workloads
    AdaptiveModelWrapper: Wrapper for ML models with adaptive behavior

Examples:
    >>> from power_management import PowerConstraint, PowerState
    >>> from adaptive_algorithms import AdaptiveController, AdaptiveModelWrapper
    >>>
    >>> constraints = PowerConstraint()
    >>> controller = AdaptiveController(constraints)
    >>>
    >>> power_state = PowerState(
    ...     cpu_usage_percent=75.0,
    ...     gpu_usage_percent=60.0,
    ...     cpu_temp_celsius=75.0,
    ...     gpu_temp_celsius=65.0,
    ...     cpu_power_watts=18.0,
    ...     gpu_power_watts=50.0,
    ...     timestamp=time.time()
    ... )
    >>>
    >>> params = controller.update_parameters(power_state)
    >>> print(f"Adaptive parameters: {params}")
"""
import time
import threading
from typing import Dict, List, Callable, Optional, Any, Tuple, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from src.qwen3_vl.system.power_management import PowerState, PowerConstraint
from src.qwen3_vl.system.thermal_management import ThermalManager


# Type aliases for better readability
ParameterHistory = List[Tuple[float, 'AdaptiveParameters']]
WorkloadList = List[Tuple[str, Callable]]
ModelType = TypeVar('ModelType')


@dataclass
class AdaptiveParameters:
    """
    Parameters for adaptive algorithms that control model behavior based on system constraints.

    This dataclass contains parameters that are used to adjust various aspects of model
    execution to maintain performance while staying within power and thermal limits.
    The parameters are dynamically adjusted by the AdaptiveController based on current
    system conditions.

    Attributes:
        performance_factor (float): Performance scaling factor (0.0-1.0, where 1.0 is full performance)
        batch_size_factor (float): Factor to adjust batch sizes (0.0-1.0)
        frequency_factor (float): Factor to adjust operation frequency (0.0-1.0)
        resource_allocation (float): Factor for resource allocation (0.0-1.0)
        execution_delay (float): Delay before execution in seconds (0.0+)
    """
    performance_factor: float = 1.0  # 0.0-1.0, where 1.0 is full performance
    """Performance scaling factor (0.0-1.0, where 1.0 is full performance)"""

    batch_size_factor: float = 1.0   # Factor to adjust batch sizes
    """Factor to adjust batch sizes (0.0-1.0)"""

    frequency_factor: float = 1.0    # Factor to adjust operation frequency
    """Factor to adjust operation frequency (0.0-1.0)"""

    resource_allocation: float = 1.0 # Factor for resource allocation
    """Factor for resource allocation (0.0-1.0)"""

    execution_delay: float = 0.0     # Delay before execution (seconds)
    """Delay before execution in seconds (0.0+)"""

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate all parameters are within acceptable ranges."""
        if not isinstance(self.performance_factor, (int, float)):
            raise TypeError(f"performance_factor must be a number, got {type(self.performance_factor)}")
        if not 0.0 <= self.performance_factor <= 1.0:
            raise ValueError(f"performance_factor must be between 0.0 and 1.0, got {self.performance_factor}")

        if not isinstance(self.batch_size_factor, (int, float)):
            raise TypeError(f"batch_size_factor must be a number, got {type(self.batch_size_factor)}")
        if not 0.0 <= self.batch_size_factor <= 1.0:
            raise ValueError(f"batch_size_factor must be between 0.0 and 1.0, got {self.batch_size_factor}")

        if not isinstance(self.frequency_factor, (int, float)):
            raise TypeError(f"frequency_factor must be a number, got {type(self.frequency_factor)}")
        if not 0.0 <= self.frequency_factor <= 1.0:
            raise ValueError(f"frequency_factor must be between 0.0 and 1.0, got {self.frequency_factor}")

        if not isinstance(self.resource_allocation, (int, float)):
            raise TypeError(f"resource_allocation must be a number, got {type(self.resource_allocation)}")
        if not 0.0 <= self.resource_allocation <= 1.0:
            raise ValueError(f"resource_allocation must be between 0.0 and 1.0, got {self.resource_allocation}")

        if not isinstance(self.execution_delay, (int, float)):
            raise TypeError(f"execution_delay must be a number, got {type(self.execution_delay)}")
        if self.execution_delay < 0.0:
            raise ValueError(f"execution_delay must be non-negative, got {self.execution_delay}")


class AdaptationStrategy(Enum):
    """
    Strategies for adaptation that determine how the system responds to constraints.

    This enum defines different approaches for balancing performance, power efficiency,
    and thermal management when system constraints are approached or exceeded.

    The strategies provide different trade-offs between performance and system health:
    - PERFORMANCE_FIRST: Minimizes performance impact while staying within constraints
    - POWER_EFFICIENT: Prioritizes power efficiency over performance
    - THERMAL_AWARE: Prioritizes thermal management to prevent overheating
    - BALANCED: Balanced approach considering all constraints equally

    Attributes:
        PERFORMANCE_FIRST: Minimize performance impact while staying within constraints
        POWER_EFFICIENT: Prioritize power efficiency over performance
        THERMAL_AWARE: Prioritize thermal management
        BALANCED: Balanced approach considering all constraints
    """
    PERFORMANCE_FIRST = "performance_first"
    """Minimize performance impact while staying within constraints."""

    POWER_EFFICIENT = "power_efficient"
    """Prioritize power efficiency over performance."""

    THERMAL_AWARE = "thermal_aware"
    """Prioritize thermal management."""

    BALANCED = "balanced"
    """Balanced approach considering all constraints."""


class AdaptiveController:
    """
    Adaptive controller that adjusts system behavior based on power and thermal constraints.

    This controller implements algorithms that can dynamically modify performance parameters
    based on current system constraints such as power consumption, temperature, and resource usage.
    It continuously monitors system state and adjusts parameters to maintain optimal performance
    while staying within defined constraints.

    The controller supports multiple adaptation strategies and maintains historical data
    for trend analysis and optimization. It provides thread-safe operations for use in
    concurrent environments.

    Key features:
    - Dynamic parameter adjustment based on system constraints
    - Multiple adaptation strategies (performance, power, thermal, balanced)
    - Historical trend analysis
    - Thread-safe operations
    - Automatic parameter adjustment loop

    Attributes:
        constraints (PowerConstraint): Power and thermal constraints for the system
        current_parameters (AdaptiveParameters): Current adaptive parameters being used
        adaptation_strategy (AdaptationStrategy): Current strategy for adaptation
        is_active (bool): Whether the controller is actively monitoring
        monitoring_thread (Optional[threading.Thread]): Thread for monitoring loop
        parameter_history (ParameterHistory): Historical record of adaptive parameters
        max_history_size (int): Maximum number of historical entries to keep
        logger (logging.Logger): Logger instance for the controller
    """

    def __init__(self, constraints: PowerConstraint) -> None:
        """
        Initialize the adaptive controller.

        Args:
            constraints: Power and thermal constraints for the system
        """
        if not isinstance(constraints, PowerConstraint):
            raise TypeError(f"constraints must be a PowerConstraint instance, got {type(constraints)}")

        self.constraints = constraints
        self.current_parameters = AdaptiveParameters()
        self.adaptation_strategy = AdaptationStrategy.BALANCED
        self.is_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.parameter_history: ParameterHistory = []  # (timestamp, params)
        self.max_history_size = 100

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def update_parameters(self, power_state: PowerState) -> AdaptiveParameters:
        """
        Update adaptive parameters based on current power state.

        This method calculates new adaptive parameters based on the current system
        state and the selected adaptation strategy. It evaluates various system
        metrics (CPU/GPU usage, temperature, power consumption) against defined
        constraints and adjusts parameters accordingly.

        The parameter adjustment process:
        1. Calculates constraint ratios for all monitored metrics
        2. Determines the maximum constraint violation
        3. Applies strategy-specific adjustments
        4. Ensures parameters remain within valid ranges
        5. Updates historical records

        Args:
            power_state: Current power and thermal state of the system

        Returns:
            Updated adaptive parameters
        """
        # Validate input
        if not isinstance(power_state, PowerState):
            raise TypeError(f"power_state must be a PowerState instance, got {type(power_state)}")

        # Validate that all values in power_state are valid
        if not isinstance(power_state.cpu_usage_percent, (int, float)) or not 0 <= power_state.cpu_usage_percent <= 100:
            raise ValueError(f"cpu_usage_percent must be between 0 and 100, got {power_state.cpu_usage_percent}")

        if not isinstance(power_state.gpu_usage_percent, (int, float)) or not 0 <= power_state.gpu_usage_percent <= 100:
            raise ValueError(f"gpu_usage_percent must be between 0 and 100, got {power_state.gpu_usage_percent}")

        if not isinstance(power_state.cpu_temp_celsius, (int, float)) or power_state.cpu_temp_celsius < -273.15:
            raise ValueError(f"cpu_temp_celsius must be a valid temperature (>= -273.15), got {power_state.cpu_temp_celsius}")

        if not isinstance(power_state.gpu_temp_celsius, (int, float)) or power_state.gpu_temp_celsius < -273.15:
            raise ValueError(f"gpu_temp_celsius must be a valid temperature (>= -273.15), got {power_state.gpu_temp_celsius}")

        if not isinstance(power_state.cpu_power_watts, (int, float)) or power_state.cpu_power_watts < 0:
            raise ValueError(f"cpu_power_watts must be non-negative, got {power_state.cpu_power_watts}")

        if not isinstance(power_state.gpu_power_watts, (int, float)) or power_state.gpu_power_watts < 0:
            raise ValueError(f"gpu_power_watts must be non-negative, got {power_state.gpu_power_watts}")

        if not isinstance(power_state.timestamp, (int, float)):
            raise TypeError(f"timestamp must be a number, got {type(power_state.timestamp)}")

        new_params = AdaptiveParameters()

        # Calculate constraint ratios
        cpu_usage_ratio = power_state.cpu_usage_percent / 100.0
        gpu_usage_ratio = power_state.gpu_usage_percent / 100.0
        cpu_temp_ratio = power_state.cpu_temp_celsius / self.constraints.max_cpu_temp_celsius
        gpu_temp_ratio = power_state.gpu_temp_celsius / self.constraints.max_gpu_temp_celsius
        cpu_power_ratio = power_state.cpu_power_watts / self.constraints.max_cpu_power_watts
        gpu_power_ratio = power_state.gpu_power_watts / self.constraints.max_gpu_power_watts

        # Validate that constraint ratios are valid
        ratios = [cpu_usage_ratio, gpu_usage_ratio, cpu_temp_ratio, gpu_temp_ratio, cpu_power_ratio, gpu_power_ratio]
        for i, ratio in enumerate(ratios):
            if not isinstance(ratio, (int, float)) or ratio < 0:
                self.logger.warning(f"Invalid constraint ratio at index {i}: {ratio}, setting to 0")
                ratios[i] = 0.0

        # Determine the maximum constraint ratio
        max_constraint = max(ratios)

        # Apply different strategies based on current strategy
        if self.adaptation_strategy == AdaptationStrategy.PERFORMANCE_FIRST:
            # Minimize performance impact while staying within constraints
            new_params.performance_factor = max(0.7, 1.0 - (max_constraint - 0.8) * 2) if max_constraint > 0.8 else 1.0
            new_params.batch_size_factor = max(0.6, 1.0 - (max_constraint - 0.8) * 1.5) if max_constraint > 0.8 else 1.0
            new_params.frequency_factor = max(0.5, 1.0 - (max_constraint - 0.8)) if max_constraint > 0.8 else 1.0

        elif self.adaptation_strategy == AdaptationStrategy.POWER_EFFICIENT:
            # Prioritize power efficiency over performance
            new_params.performance_factor = max(0.3, 1.0 - max_constraint)
            new_params.batch_size_factor = max(0.3, 1.0 - max_constraint * 0.8)
            new_params.frequency_factor = max(0.2, 1.0 - max_constraint * 0.9)

        elif self.adaptation_strategy == AdaptationStrategy.THERMAL_AWARE:
            # Prioritize thermal management
            max_temp_ratio = max(cpu_temp_ratio, gpu_temp_ratio)
            new_params.performance_factor = max(0.4, 1.0 - max_temp_ratio)
            new_params.batch_size_factor = max(0.4, 1.0 - max_temp_ratio * 0.7)
            new_params.frequency_factor = max(0.3, 1.0 - max_temp_ratio * 0.8)
            # Add execution delay if thermal stress is high
            if max_temp_ratio > 0.9:
                new_params.execution_delay = 0.1  # 100ms delay

        else:  # BALANCED
            # Balanced approach considering all constraints
            new_params.performance_factor = max(0.4, 1.0 - max_constraint * 0.7)
            new_params.batch_size_factor = max(0.4, 1.0 - max_constraint * 0.6)
            new_params.frequency_factor = max(0.3, 1.0 - max_constraint * 0.8)

            # Adjust based on individual constraint violations
            if cpu_temp_ratio > 0.9 or gpu_temp_ratio > 0.9:
                new_params.performance_factor *= 0.8
                new_params.execution_delay = 0.05  # 50ms delay

            if cpu_power_ratio > 0.9 or gpu_power_ratio > 0.9:
                new_params.batch_size_factor *= 0.85

        # Ensure parameters are within valid ranges
        new_params.performance_factor = max(0.1, min(1.0, new_params.performance_factor))
        new_params.batch_size_factor = max(0.1, min(1.0, new_params.batch_size_factor))
        new_params.frequency_factor = max(0.1, min(1.0, new_params.frequency_factor))
        new_params.resource_allocation = new_params.performance_factor
        new_params.execution_delay = max(0.0, min(0.5, new_params.execution_delay))

        # Store in history
        self._add_to_history(power_state.timestamp, new_params)

        self.current_parameters = new_params
        return new_params

    def _add_to_history(self, timestamp: float, params: AdaptiveParameters) -> None:
        """
        Add parameters to history, maintaining max history size.

        This method adds the current parameter set to the historical record,
        ensuring the history doesn't exceed the maximum allowed size by
        removing the oldest entries when necessary.

        Args:
            timestamp: Time when parameters were recorded
            params: Adaptive parameters to store in history
        """
        # Validate inputs
        if not isinstance(timestamp, (int, float)):
            raise TypeError(f"timestamp must be a number, got {type(timestamp)}")
        if not isinstance(params, AdaptiveParameters):
            raise TypeError(f"params must be an AdaptiveParameters instance, got {type(params)}")

        self.parameter_history.append((timestamp, params))
        if len(self.parameter_history) > self.max_history_size:
            self.parameter_history.pop(0)

    def get_historical_trend(self, window_size: int = 10) -> Optional[Dict[str, float]]:
        """
        Get trend of parameters over time.

        Calculates trends for performance, batch size, and frequency factors
        over the specified window size using linear regression analysis.
        This information can be used to predict future system behavior
        and adjust parameters proactively.

        Args:
            window_size: Number of recent entries to consider for trend calculation

        Returns:
            Dictionary containing trend information or None if insufficient data
        """
        # Validate input
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")

        if len(self.parameter_history) < 2:
            return None

        recent_history = self.parameter_history[-window_size:]
        if not recent_history:
            return None

        # Calculate trends
        try:
            performance_values = [p.performance_factor for _, p in recent_history]
            batch_values = [p.batch_size_factor for _, p in recent_history]
            freq_values = [p.frequency_factor for _, p in recent_history]
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Error extracting parameter values from history: {e}")
            return None

        try:
            return {
                'performance_trend': self._calculate_trend(performance_values),
                'batch_trend': self._calculate_trend(batch_values),
                'frequency_trend': self._calculate_trend(freq_values),
                'avg_performance': sum(performance_values) / len(performance_values),
                'avg_batch': sum(batch_values) / len(batch_values),
                'avg_frequency': sum(freq_values) / len(freq_values)
            }
        except ZeroDivisionError:
            self.logger.error("Error calculating averages due to empty values list")
            return None

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend using linear regression.

        This method performs simple linear regression on a series of values
        to determine the trend direction and magnitude. A positive slope
        indicates an increasing trend, while a negative slope indicates
        a decreasing trend.

        Args:
            values: List of numerical values to calculate trend for

        Returns:
            Slope of the linear regression line (trend)
        """
        # Validate input
        if not isinstance(values, list):
            raise TypeError(f"values must be a list, got {type(values)}")

        if len(values) < 2:
            return 0.0

        # Validate all values are numbers
        for i, val in enumerate(values):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Value at index {i} must be a number, got {type(val)}: {val}")

        n = len(values)
        x = list(range(n))
        y = values

        try:
            # Simple linear regression to find trend
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)

            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
        except (ZeroDivisionError, OverflowError) as e:
            self.logger.error(f"Error calculating trend: {e}")
            return 0.0

    def adapt_model_behavior(self, model_func: Callable, *args, **kwargs) -> Any:
        """
        Adapt model behavior based on current parameters.

        Applies current adaptive parameters to modify model execution,
        including adjusting batch sizes, adding delays, and simulating
        frequency adjustments. This method serves as a wrapper that
        transparently applies adaptive modifications to model execution.

        The adaptation process:
        1. Applies execution delay if specified
        2. Adjusts batch size based on current factor
        3. Simulates frequency adjustments (in real implementation)
        4. Executes the wrapped function with adjusted parameters

        Args:
            model_func: The model function to execute
            *args: Positional arguments to pass to the model function
            **kwargs: Keyword arguments to pass to the model function

        Returns:
            Result of the model function execution
        """
        # Validate inputs
        if not callable(model_func):
            raise TypeError(f"model_func must be callable, got {type(model_func)}")

        # Apply parameter adjustments
        try:
            if self.current_parameters.execution_delay > 0:
                time.sleep(self.current_parameters.execution_delay)
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Invalid execution delay value: {e}")

        # Adjust batch size if present in kwargs
        if 'batch_size' in kwargs:
            try:
                original_batch_size = kwargs['batch_size']
                if not isinstance(original_batch_size, (int, float)):
                    raise TypeError(f"batch_size must be a number, got {type(original_batch_size)}")

                adjusted_batch_size = int(original_batch_size * self.current_parameters.batch_size_factor)
                kwargs['batch_size'] = max(1, adjusted_batch_size)  # Ensure at least 1
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Error adjusting batch size: {e}")

        # Adjust frequency by skipping some operations (simulated)
        if self.current_parameters.frequency_factor < 1.0:
            # In a real implementation, this might skip certain operations
            # based on the frequency factor
            pass

        # Call the model function with adjusted parameters
        try:
            return model_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error executing model function: {e}")
            raise

    def start_adaptation(self, monitoring_interval: float = 1.0) -> None:
        """
        Start adaptive parameter adjustment.

        Begins the monitoring loop that periodically updates adaptive parameters
        based on system state. The monitoring runs in a separate thread to
        avoid blocking the main application flow.

        Args:
            monitoring_interval: Time interval between parameter updates in seconds
        """
        # Validate input
        if not isinstance(monitoring_interval, (int, float)) or monitoring_interval <= 0:
            raise ValueError(f"monitoring_interval must be a positive number, got {monitoring_interval}")

        if self.is_active:
            self.logger.warning("Adaptation is already active")
            return

        self.is_active = True
        self.monitoring_interval = monitoring_interval

        try:
            self.monitoring_thread = threading.Thread(
                target=self._adaptation_loop,
                args=(monitoring_interval,)
            )
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

            self.logger.info("Started adaptive parameter adjustment")
        except Exception as e:
            self.logger.error(f"Error starting adaptation thread: {e}")
            self.is_active = False
            raise

    def stop_adaptation(self) -> None:
        """
        Stop adaptive parameter adjustment.

        Stops the monitoring loop and waits for it to finish. This method
        ensures clean shutdown of the adaptive system by properly terminating
        the monitoring thread.
        """
        self.is_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            try:
                self.monitoring_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Error joining adaptation thread: {e}")

        self.logger.info("Stopped adaptive parameter adjustment")

    def _adaptation_loop(self, interval: float) -> None:
        """
        Main adaptation loop.

        This would typically be called from an external monitoring system
        to continuously update parameters based on system state. The loop
        runs at the specified interval and updates parameters accordingly.

        Args:
            interval: Time interval between adaptation checks
        """
        # This would typically be called from an external monitoring system
        # For this implementation, we'll just log that the loop is running
        self.logger.info("Adaptive controller running...")

    def set_strategy(self, strategy: AdaptationStrategy) -> None:
        """
        Set adaptation strategy.

        Changes the current adaptation strategy which affects how parameters
        are adjusted based on system constraints. Different strategies provide
        different trade-offs between performance and system health.

        Args:
            strategy: The new adaptation strategy to use
        """
        if not isinstance(strategy, AdaptationStrategy):
            raise TypeError(f"strategy must be an AdaptationStrategy, got {type(strategy)}")

        self.adaptation_strategy = strategy
        self.logger.info(f"Adaptation strategy set to {strategy.value}")

    def get_current_parameters(self) -> AdaptiveParameters:
        """
        Get current adaptive parameters.

        Returns the current adaptive parameters being used by the controller.
        These parameters reflect the most recent system state evaluation.

        Returns:
            Current adaptive parameters being used by the controller
        """
        return self.current_parameters

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """
        Get summary of adaptation status.

        Provides a comprehensive overview of the current adaptation state,
        including parameters, strategy, activity status, and historical trends.
        This information is useful for monitoring and debugging the adaptive system.

        Returns:
            Dictionary containing current parameters, strategy, activity status,
            history size, and trend information
        """
        try:
            trend = self.get_historical_trend()
        except Exception as e:
            self.logger.error(f"Error getting historical trend: {e}")
            trend = None

        try:
            return {
                "current_parameters": {
                    "performance_factor": self.current_parameters.performance_factor,
                    "batch_size_factor": self.current_parameters.batch_size_factor,
                    "frequency_factor": self.current_parameters.frequency_factor,
                    "resource_allocation": self.current_parameters.resource_allocation,
                    "execution_delay": self.current_parameters.execution_delay
                },
                "strategy": self.adaptation_strategy.value,
                "active": self.is_active,
                "history_size": len(self.parameter_history),
                "trend": trend
            }
        except Exception as e:
            self.logger.error(f"Error creating adaptation summary: {e}")
            raise


class LoadBalancer:
    """
    Adaptive load balancer that distributes work based on power and thermal constraints.

    This class manages the distribution of workloads across available resources
    based on current system constraints to optimize performance while staying
    within power and thermal limits. It dynamically adjusts the distribution
    of work based on real-time system monitoring.

    The load balancer considers multiple system metrics (CPU/GPU usage, temperature)
    and adjusts workload distribution accordingly to prevent system overload
    while maintaining optimal performance.

    Key features:
    - Dynamic workload distribution based on system constraints
    - Historical tracking of load distribution
    - Thread-safe operations
    - Error handling for workload execution

    Attributes:
        constraints (PowerConstraint): Power and thermal constraints for the system
        work_distribution (Dict[str, float]): Current distribution factors for workloads
        load_history (Dict[str, List[Tuple[float, float]]]): Historical record of load distribution
        max_history_size (int): Maximum number of historical entries to keep
        logger (logging.Logger): Logger instance for the load balancer
    """

    def __init__(self, constraints: PowerConstraint) -> None:
        """
        Initialize the load balancer.

        Args:
            constraints: Power and thermal constraints for the system
        """
        if not isinstance(constraints, PowerConstraint):
            raise TypeError(f"constraints must be a PowerConstraint instance, got {type(constraints)}")

        self.constraints = constraints
        self.work_distribution: Dict[str, float] = {}  # workload_id: distribution_factor
        self.load_history: Dict[str, List[Tuple[float, float]]] = {}  # workload_id: [(timestamp, load), ...]
        self.max_history_size = 50

        # Initialize logging
        self.logger = logging.getLogger(__name__)

    def distribute_load(self, workloads: WorkloadList, power_state: PowerState) -> Dict[str, float]:
        """
        Distribute load among workloads based on power state.

        Calculates distribution factors for each workload based on current
        system constraints to ensure optimal resource utilization while
        staying within defined limits.

        The distribution algorithm:
        1. Evaluates current system constraints (CPU/GPU usage, temperature)
        2. Determines the primary constraint limiting system performance
        3. Calculates appropriate distribution factors based on constraint severity
        4. Maintains historical records for trend analysis

        Args:
            workloads: List of (workload_id, workload_function) tuples
            power_state: Current power and thermal state of the system

        Returns:
            Dictionary mapping workload IDs to their distribution factors
        """
        # Validate inputs
        if not isinstance(workloads, list):
            raise TypeError(f"workloads must be a list, got {type(workloads)}")

        if not isinstance(power_state, PowerState):
            raise TypeError(f"power_state must be a PowerState instance, got {type(power_state)}")

        # Validate workloads content
        for i, workload in enumerate(workloads):
            if not isinstance(workload, tuple) or len(workload) != 2:
                raise ValueError(f"Workload at index {i} must be a (workload_id, workload_function) tuple, got {workload}")

            workload_id, workload_func = workload
            if not isinstance(workload_id, str):
                raise TypeError(f"Workload ID at index {i} must be a string, got {type(workload_id)}")

            if not callable(workload_func):
                raise TypeError(f"Workload function at index {i} must be callable, got {type(workload_func)}")

        # Validate that constraints are valid
        if self.constraints.max_cpu_temp_celsius <= 0 or self.constraints.max_gpu_temp_celsius <= 0:
            raise ValueError("Max temperature constraints must be positive")

        if self.constraints.max_cpu_power_watts <= 0 or self.constraints.max_gpu_power_watts <= 0:
            raise ValueError("Max power constraints must be positive")

        try:
            # Calculate constraint ratios
            cpu_usage_ratio = power_state.cpu_usage_percent / 100.0
            gpu_usage_ratio = power_state.gpu_usage_percent / 100.0
            cpu_temp_ratio = power_state.cpu_temp_celsius / self.constraints.max_cpu_temp_celsius
            gpu_temp_ratio = power_state.gpu_temp_celsius / self.constraints.max_gpu_temp_celsius

            # Validate ratios are valid
            ratios = [cpu_usage_ratio, gpu_usage_ratio, cpu_temp_ratio, gpu_temp_ratio]
            for i, ratio in enumerate(ratios):
                if not isinstance(ratio, (int, float)) or ratio < 0:
                    self.logger.warning(f"Invalid constraint ratio at index {i}: {ratio}, setting to 0")
                    ratios[i] = 0.0

            # Determine primary constraint
            primary_constraint = max(ratios)

            # Calculate distribution factors
            distribution_factors: Dict[str, float] = {}
            total_workloads = len(workloads)

            if total_workloads == 0:
                return distribution_factors  # Return empty dict if no workloads

            if primary_constraint > 0.9:
                # Severe constraint - distribute very conservatively
                for workload_id, _ in workloads:
                    distribution_factors[workload_id] = max(0.1, 0.3 / total_workloads)
            elif primary_constraint > 0.8:
                # Moderate constraint - reduce load
                for workload_id, _ in workloads:
                    distribution_factors[workload_id] = max(0.15, 0.6 / total_workloads)
            elif primary_constraint > 0.6:
                # Mild constraint - slight reduction
                for workload_id, _ in workloads:
                    distribution_factors[workload_id] = max(0.2, 0.85 / total_workloads)
            else:
                # Good conditions - near full load
                for workload_id, _ in workloads:
                    distribution_factors[workload_id] = min(1.0, 0.95 / total_workloads)

            # Store distribution in history
            timestamp = time.time()
            for workload_id, _ in workloads:
                if workload_id not in self.load_history:
                    self.load_history[workload_id] = []
                self.load_history[workload_id].append((timestamp, distribution_factors[workload_id]))

                # Maintain history size
                if len(self.load_history[workload_id]) > self.max_history_size:
                    self.load_history[workload_id].pop(0)

            self.work_distribution = distribution_factors
            return distribution_factors
        except Exception as e:
            self.logger.error(f"Error in distribute_load: {e}")
            raise

    def execute_workloads(self, workloads: WorkloadList, power_state: PowerState) -> Dict[str, Any]:
        """
        Execute workloads with adaptive load distribution.

        Distributes the workloads based on current system constraints and
        executes them with appropriate load factors. This method combines
        load distribution with workload execution to provide a complete
        adaptive execution pipeline.

        The execution process:
        1. Distributes workloads based on current system state
        2. Executes each workload with its assigned factor
        3. Handles errors gracefully
        4. Returns comprehensive results

        Args:
            workloads: List of (workload_id, workload_function) tuples
            power_state: Current power and thermal state of the system

        Returns:
            Dictionary containing results for each workload
        """
        # Validate inputs
        if not isinstance(workloads, list):
            raise TypeError(f"workloads must be a list, got {type(workloads)}")

        if not isinstance(power_state, PowerState):
            raise TypeError(f"power_state must be a PowerState instance, got {type(power_state)}")

        try:
            distribution_factors = self.distribute_load(workloads, power_state)
        except Exception as e:
            self.logger.error(f"Error in workload distribution: {e}")
            # Create default distribution factors if distribution fails
            distribution_factors = {workload_id: 1.0 for workload_id, _ in workloads}

        results: Dict[str, Any] = {}

        for workload_id, workload_func in workloads:
            factor = distribution_factors.get(workload_id, 1.0)

            # Apply load factor by adjusting execution parameters
            # This is a simplified example - in practice, this would adjust
            # parameters like batch size, number of iterations, etc.
            try:
                # In a real implementation, we might pass the factor to the workload
                # or adjust execution parameters based on the factor
                result = workload_func(factor)
                results[workload_id] = {
                    'result': result,
                    'factor': factor,
                    'status': 'success'
                }
            except Exception as e:
                self.logger.error(f"Error executing workload {workload_id}: {e}")
                results[workload_id] = {
                    'result': None,
                    'factor': factor,
                    'status': 'error',
                    'error': str(e)
                }

        return results


class AdaptiveModelWrapper:
    """
    Wrapper for machine learning models that adapts behavior based on power/thermal constraints.

    This wrapper provides an interface that automatically adjusts model execution
    parameters based on current system constraints to optimize performance while
    staying within power and thermal limits. It integrates with the adaptive
    controller and load balancer to provide comprehensive adaptation capabilities.

    The wrapper automatically adjusts various aspects of model execution including
    batch sizes, performance scaling, and execution delays based on real-time
    system monitoring. It supports both prediction and training operations with
    adaptive behavior.

    Key features:
    - Automatic parameter adjustment based on system constraints
    - Support for both prediction and training with adaptation
    - Integration with adaptive controller and load balancer
    - Simulation capabilities for testing and development

    Attributes:
        model (Any): The underlying machine learning model
        constraints (PowerConstraint): Power and thermal constraints for the system
        adaptive_controller (AdaptiveController): Controller for adaptive parameter management
        load_balancer (LoadBalancer): Load balancer for workload distribution
        logger (logging.Logger): Logger instance for the wrapper
    """

    def __init__(self, model: Any, constraints: PowerConstraint) -> None:
        """
        Initialize the adaptive model wrapper.

        Args:
            model: The underlying machine learning model to wrap
            constraints: Power and thermal constraints for the system
        """
        if not isinstance(constraints, PowerConstraint):
            raise TypeError(f"constraints must be a PowerConstraint instance, got {type(constraints)}")

        self.model = model
        self.constraints = constraints

        try:
            self.adaptive_controller = AdaptiveController(constraints)
            self.load_balancer = LoadBalancer(constraints)
        except Exception as e:
            self.logger.error(f"Error initializing adaptive components: {e}")
            raise

        # Initialize logging
        self.logger = logging.getLogger(__name__)

    def predict(self, input_data: Any, power_state: PowerState, **kwargs) -> Dict[str, Any]:
        """
        Adaptive prediction that adjusts based on power state.

        Updates adaptive parameters based on the current power state and
        adjusts prediction parameters accordingly before executing the prediction.
        This method provides a seamless interface that automatically adapts
        model execution to current system conditions.

        The adaptation process:
        1. Updates adaptive parameters based on current power state
        2. Adjusts prediction parameters (batch size, etc.) based on constraints
        3. Applies performance scaling and delays as needed
        4. Executes the prediction with adaptive parameters

        Args:
            input_data: Input data for the prediction
            power_state: Current power and thermal state of the system
            **kwargs: Additional arguments for the prediction

        Returns:
            Dictionary containing prediction result and parameters used
        """
        # Validate inputs
        if not isinstance(power_state, PowerState):
            raise TypeError(f"power_state must be a PowerState instance, got {type(power_state)}")

        try:
            # Update adaptive parameters based on power state
            params = self.adaptive_controller.update_parameters(power_state)
        except Exception as e:
            self.logger.error(f"Error updating adaptive parameters: {e}")
            # Use default parameters if update fails
            params = AdaptiveParameters()

        # Adjust prediction parameters based on constraints
        if 'batch_size' in kwargs:
            try:
                original_batch_size = kwargs['batch_size']
                if not isinstance(original_batch_size, (int, float)):
                    raise TypeError(f"batch_size must be a number, got {type(original_batch_size)}")

                adjusted_batch_size = int(original_batch_size * params.batch_size_factor)
                kwargs['batch_size'] = max(1, adjusted_batch_size)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Error adjusting batch size: {e}")

        # Apply performance factor by potentially skipping some computations
        # In a real implementation, this might involve early stopping,
        # reducing precision, or skipping layers
        if params.performance_factor < 1.0:
            # This is a simplified example - in practice, this would adjust
            # the model's execution based on the performance factor
            pass

        # Execute prediction with adjusted parameters
        try:
            if params.execution_delay > 0:
                time.sleep(params.execution_delay)
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Invalid execution delay value: {e}")

        # In a real implementation, we would call the actual model prediction
        # For this example, we'll simulate a prediction
        try:
            return self._simulate_prediction(input_data, params)
        except Exception as e:
            self.logger.error(f"Error in prediction simulation: {e}")
            raise

    def _simulate_prediction(self, input_data: Any, params: AdaptiveParameters) -> Dict[str, Any]:
        """
        Simulate a prediction with adaptive parameters.

        This method simulates model prediction behavior with adaptive parameters
        for testing and development purposes. In a real implementation, this
        would call the actual model prediction method.

        Args:
            input_data: Input data for the prediction
            params: Adaptive parameters to use for the simulation

        Returns:
            Dictionary containing simulated prediction result and parameters used
        """
        # Simulate processing time based on performance factor
        base_time = 0.1  # Base processing time in seconds
        adjusted_time = base_time / params.performance_factor
        time.sleep(min(adjusted_time, 1.0))  # Cap at 1 second

        # Simulate a result (in practice, this would be the model's output)
        result = {
            'prediction': np.random.random((len(input_data), 10)).tolist() if hasattr(input_data, '__len__') else [0.5],
            'parameters_used': {
                'performance_factor': params.performance_factor,
                'batch_size_factor': params.batch_size_factor,
                'frequency_factor': params.frequency_factor
            }
        }

        return result

    def fit(self, training_data: Any, power_state: PowerState, **kwargs) -> Dict[str, Any]:
        """
        Adaptive training that adjusts based on power state.

        Updates adaptive parameters based on the current power state and
        adjusts training parameters accordingly before executing the training.
        This method provides adaptive training capabilities that respond to
        system constraints in real-time.

        The adaptive training process:
        1. Updates adaptive parameters based on current power state
        2. Adjusts training parameters (epochs, batch size, etc.) based on constraints
        3. Applies performance scaling and delays as needed
        4. Executes the training with adaptive parameters

        Args:
            training_data: Training data for the model
            power_state: Current power and thermal state of the system
            **kwargs: Additional arguments for the training

        Returns:
            Dictionary containing training result and parameters used
        """
        # Validate inputs
        if not isinstance(power_state, PowerState):
            raise TypeError(f"power_state must be a PowerState instance, got {type(power_state)}")

        try:
            # Update adaptive parameters based on power state
            params = self.adaptive_controller.update_parameters(power_state)
        except Exception as e:
            self.logger.error(f"Error updating adaptive parameters: {e}")
            # Use default parameters if update fails
            params = AdaptiveParameters()

        # Adjust training parameters based on constraints
        if 'epochs' in kwargs:
            try:
                original_epochs = kwargs['epochs']
                if not isinstance(original_epochs, (int, float)):
                    raise TypeError(f"epochs must be a number, got {type(original_epochs)}")

                adjusted_epochs = int(original_epochs * params.performance_factor)
                kwargs['epochs'] = max(1, adjusted_epochs)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Error adjusting epochs: {e}")

        if 'batch_size' in kwargs:
            try:
                original_batch_size = kwargs['batch_size']
                if not isinstance(original_batch_size, (int, float)):
                    raise TypeError(f"batch_size must be a number, got {type(original_batch_size)}")

                adjusted_batch_size = int(original_batch_size * params.batch_size_factor)
                kwargs['batch_size'] = max(1, adjusted_batch_size)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Error adjusting batch size: {e}")

        # Apply frequency factor by potentially reducing training frequency
        if params.frequency_factor < 1.0:
            # This might involve reducing the number of updates per epoch
            pass

        # Execute training with adjusted parameters
        try:
            if params.execution_delay > 0:
                time.sleep(params.execution_delay)
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Invalid execution delay value: {e}")

        # In a real implementation, we would call the actual model training
        # For this example, we'll simulate training
        try:
            return self._simulate_training(training_data, params)
        except Exception as e:
            self.logger.error(f"Error in training simulation: {e}")
            raise

    def _simulate_training(self, training_data: Any, params: AdaptiveParameters) -> Dict[str, Any]:
        """
        Simulate training with adaptive parameters.

        This method simulates model training behavior with adaptive parameters
        for testing and development purposes. In a real implementation, this
        would call the actual model training method.

        Args:
            training_data: Training data for the model
            params: Adaptive parameters to use for the simulation

        Returns:
            Dictionary containing simulated training result and parameters used
        """
        # Simulate training time based on performance factor
        base_time = 1.0  # Base training time in seconds
        adjusted_time = base_time / params.performance_factor
        time.sleep(min(adjusted_time, 5.0))  # Cap at 5 seconds

        # Simulate a result (in practice, this would be training metrics)
        result = {
            'final_loss': np.random.random(),
            'epochs_trained': int(10 * params.performance_factor),
            'parameters_used': {
                'performance_factor': params.performance_factor,
                'batch_size_factor': params.batch_size_factor,
                'frequency_factor': params.frequency_factor
            }
        }

        return result


if __name__ == "__main__":
    # Example usage
    constraints = PowerConstraint()
    controller = AdaptiveController(constraints)
    
    # Simulate power states
    power_state = PowerState(
        cpu_usage_percent=75.0,
        gpu_usage_percent=60.0,
        cpu_temp_celsius=75.0,
        gpu_temp_celsius=65.0,
        cpu_power_watts=18.0,
        gpu_power_watts=50.0,
        timestamp=time.time()
    )
    
    # Update parameters based on power state
    params = controller.update_parameters(power_state)
    print(f"Adaptive parameters: {params}")
    
    # Get adaptation summary
    summary = controller.get_adaptation_summary()
    print(f"Adaptation summary: {summary}")
    
    # Example of using AdaptiveModelWrapper
    class DummyModel:
        def predict(self, X):
            return [0.5] * len(X) if hasattr(X, '__len__') else [0.5]
    
    dummy_model = DummyModel()
    adaptive_model = AdaptiveModelWrapper(dummy_model, constraints)
    
    # Simulate prediction with adaptive parameters
    input_data = [1, 2, 3, 4, 5]
    result = adaptive_model.predict(input_data, power_state)
    print(f"Adaptive prediction result: {result}")