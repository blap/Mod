"""
Utilities for benchmarking Qwen3-VL model performance
"""
import time
import torch
import psutil
from contextlib import contextmanager
from typing import Dict, Callable, Any, Optional
import numpy as np


@contextmanager
def timer():
    """Context manager to time execution."""
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage."""
    # CPU memory
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        start_gpu_memory = 0
    
    yield
    
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    if torch.cuda.is_available():
        end_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        gpu_peak = torch.cuda.max_memory_reserved() / 1024 / 1024  # MB
    else:
        end_gpu_memory = 0
        gpu_peak = 0
    
    print(f"CPU Memory change: {end_memory - start_memory:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory change: {end_gpu_memory - start_gpu_memory:.2f} MB")
        print(f"GPU Peak Memory: {gpu_peak:.2f} MB")


def benchmark_model_inference(
    model: torch.nn.Module,
    input_data: Dict[str, torch.Tensor],
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Benchmark model inference performance.
    
    Args:
        model: The model to benchmark
        input_data: Input data dictionary
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs
        device: Device to run benchmark on
    
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    
    # Move model and inputs to device
    if device:
        model = model.to(device)
        input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(**input_data)
    
    # Actual benchmark runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(**input_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'avg_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'num_runs': num_runs,
        'device': str(device) if device else str(next(model.parameters()).device),
        'throughput_samples_per_sec': float(num_runs / np.sum(times))
    }
    
    return results


def benchmark_multimodal_task(
    model: torch.nn.Module,
    text_input: torch.Tensor,
    image_input: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark multimodal task performance.
    
    Args:
        model: The multimodal model to benchmark
        text_input: Text input tensor
        image_input: Image input tensor
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    
    # Determine device
    device = next(model.parameters()).device
    
    # Move inputs to device
    text_input = text_input.to(device)
    image_input = image_input.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_ids=text_input, pixel_values=image_input)
    
    # Actual benchmark runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_ids=text_input, pixel_values=image_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'avg_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'num_runs': num_runs,
        'throughput_samples_per_sec': float(num_runs / np.sum(times)),
        'device': str(device)
    }
    
    return results


def profile_memory_usage(
    model: torch.nn.Module,
    input_data: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Profile memory usage of the model.
    
    Args:
        model: The model to profile
        input_data: Input data for the model
        device: Device to run profiling on
    
    Returns:
        Dictionary with memory usage statistics
    """
    if device:
        model = model.to(device)
        input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
    
    # Record initial memory
    initial_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        initial_gpu_memory = 0
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        _ = model(**input_data)
    
    # Record memory after forward pass
    cpu_memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    if torch.cuda.is_available():
        gpu_memory_after = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        peak_gpu_memory = torch.cuda.max_memory_reserved() / 1024 / 1024  # MB
    else:
        gpu_memory_after = 0
        peak_gpu_memory = 0
    
    results = {
        'cpu_memory_initial_mb': initial_cpu_memory,
        'cpu_memory_after_mb': cpu_memory_after,
        'cpu_memory_increase_mb': cpu_memory_after - initial_cpu_memory,
        'gpu_memory_initial_mb': initial_gpu_memory if torch.cuda.is_available() else 0,
        'gpu_memory_after_mb': gpu_memory_after if torch.cuda.is_available() else 0,
        'gpu_memory_increase_mb': gpu_memory_after - initial_gpu_memory if torch.cuda.is_available() else 0,
        'gpu_peak_memory_mb': peak_gpu_memory if torch.cuda.is_available() else 0
    }
    
    return results


def benchmark_generation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    num_runs: int = 5,
    warmup_runs: int = 2
) -> Dict[str, Any]:
    """
    Benchmark text generation performance.
    
    Args:
        model: The model to benchmark
        input_ids: Input token IDs
        max_new_tokens: Maximum new tokens to generate
        num_runs: Number of generation runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with generation benchmark results
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
    
    # Actual benchmark runs
    times = []
    tokens_generated = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=0
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        tokens_generated.append(output.shape[1] - input_ids.shape[1])  # Count new tokens
    
    # Calculate statistics
    times = np.array(times)
    tokens_generated = np.array(tokens_generated)
    
    results = {
        'avg_generation_time': float(np.mean(times)),
        'std_generation_time': float(np.std(times)),
        'avg_tokens_per_sec': float(np.mean(tokens_generated) / np.mean(times)),
        'avg_new_tokens_generated': float(np.mean(tokens_generated)),
        'num_runs': num_runs
    }
    
    return results