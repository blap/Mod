"""
Demonstration script for the feedback controller system with all 4 models.
"""

import time
import torch
from src.inference_pio.common.feedback_controller import (
    get_feedback_controller,
    PerformanceMetrics
)
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def simulate_model_performance():
    """Simulate performance metrics for demonstration purposes."""
    # Accuracy varies randomly around 0.9
    accuracy = 0.85 + (torch.rand(1).item() * 0.1)
    # Latency varies randomly around 0.05 seconds
    latency = 0.04 + (torch.rand(1).item() * 0.02)
    # Throughput varies based on latency and input size
    throughput = 100 / latency if latency > 0 else 0
    
    return PerformanceMetrics(
        accuracy=accuracy,
        latency=latency,
        throughput=throughput
    )


def demo_feedback_system():
    """Demonstrate the feedback system with all 4 models."""
    print("Starting Feedback Controller Demonstration")
    print("=" * 50)
    
    # Get the global feedback controller
    controller = get_feedback_controller()
    
    # Register all 4 models with the feedback controller
    model_ids = [
        "GLM-4-7",
        "Qwen3-4b-instruct-2507", 
        "Qwen3-coder-30b",
        "Qwen3-vl-2b"
    ]
    
    for model_id in model_ids:
        controller.register_model(model_id)
        print(f"Registered model: {model_id}")
    
    print("\nSimulating performance metrics collection...")
    
    # Simulate collecting metrics over time for each model
    for iteration in range(5):
        print(f"\nIteration {iteration + 1}/5:")
        
        for model_id in model_ids:
            # Simulate model performance
            metrics = simulate_model_performance()
            
            # Record metrics with the feedback controller
            controller.record_metrics(model_id, metrics)
            
            print(f"  {model_id}: Acc={metrics.accuracy:.3f}, Lat={metrics.latency:.3f}s, Thr={metrics.throughput:.1f} t/s")
            
            # Get and display current metrics
            current = controller.get_current_metrics(model_id)
            if current:
                print(f"    Current stored: Acc={current.accuracy:.3f}, Lat={current.latency:.3f}s")
        
        time.sleep(0.1)  # Brief pause between iterations
    
    print("\nDemonstrating historical metrics retrieval...")
    
    for model_id in model_ids:
        history = controller.get_historical_metrics(model_id, count=3)
        print(f"  {model_id} last 3 metrics: {[f'{m.accuracy:.3f}' for m in history]}")
    
    print("\nFeedback Controller Demonstration Complete!")
    print("=" * 50)


def demo_model_integration():
    """Demonstrate how models integrate with the feedback system."""
    print("\nModel Integration Demonstration")
    print("=" * 40)
    
    # Show how a model would integrate with feedback (conceptually)
    print("Models now inherit from FeedbackIntegrationMixin which:")
    print("  1. Automatically registers the model with the feedback controller")
    print("  2. Provides record_performance_metrics() method")
    print("  3. Sets up callback handlers for optimization adjustments")
    print("  4. Monitors performance during forward/generate calls")
    
    print("\nThe following methods are now available in all models:")
    print("  - _apply_precision_increase(): Increase model precision for better accuracy")
    print("  - _apply_speed_optimization(): Optimize for speed (e.g., lower precision)")
    print("  - _apply_accuracy_improvement(): Improve accuracy through various techniques")
    print("  - record_performance_metrics(): Record performance metrics")


if __name__ == "__main__":
    demo_feedback_system()
    demo_model_integration()