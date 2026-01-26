"""
Demonstration script for Dynamic Text Batching System.

This script demonstrates the capabilities of the dynamic text batching system
for unimodal models with different types of textual inputs.
"""

import time
import torch
import sys
import os

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.inference_pio.common.dynamic_text_batching import (
    get_dynamic_text_batch_manager,
    TextBatchType,
    DynamicTextBatchManager
)
from src.inference_pio.common.input_complexity_analyzer import get_complexity_analyzer


def demonstrate_dynamic_text_batching():
    """Demonstrate the dynamic text batching system with various text types."""
    print("=" * 60)
    print("DYNAMIC TEXT BATCHING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create a dynamic text batch manager
    batch_manager = get_dynamic_text_batch_manager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16,
        memory_threshold_ratio=0.80,
        complexity_weight=0.4,
        sequence_length_weight=0.3
    )
    
    complexity_analyzer = get_complexity_analyzer()
    
    print(f"Initial batch size: {batch_manager.current_batch_size}")
    print(f"Batch size range: [{batch_manager.min_batch_size}, {batch_manager.max_batch_size}]")
    print(f"Memory threshold: {batch_manager.memory_threshold_ratio}")
    print()
    
    # Define test cases with different text types
    test_cases = [
        {
            "name": "Simple Text",
            "texts": ["Hello world", "How are you?", "Nice to meet you", "Goodbye"],
            "description": "Basic conversational text with simple vocabulary"
        },
        {
            "name": "Technical Text",
            "texts": [
                "The transformer architecture utilizes self-attention mechanisms.",
                "Backpropagation adjusts weights through gradient descent.",
                "Convolutional neural networks excel at image recognition tasks.",
                "Recurrent networks handle sequential data effectively."
            ],
            "description": "Technical terminology and complex concepts"
        },
        {
            "name": "Code Snippets",
            "texts": [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "class Calculator:\n    def add(self, a, b):\n        return a + b",
                "for i in range(10):\n    print(i)",
                "if condition:\n    do_something()\nelse:\n    do_other()"
            ],
            "description": "Programming code with syntax complexity"
        },
        {
            "name": "Long Context",
            "texts": [
                "This is a very long text that contains many sentences and complex ideas. " * 20,
                "Another extended passage with detailed explanations and nuanced concepts. " * 20,
                "Yet another lengthy text with multiple paragraphs and sections. " * 20,
                "Final long text with comprehensive coverage of various topics. " * 20
            ],
            "description": "Extended texts with high token counts"
        }
    ]
    
    print("Testing different text types with dynamic batching:")
    print("-" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']} - {case['description']}")
        print(f"   Texts: {len(case['texts'])} samples")
        
        # Analyze the text batch type
        batch_type = batch_manager.analyze_text_batch_type(case['texts'])
        print(f"   Detected batch type: {batch_type.value}")
        
        # Analyze complexity
        complexity_metrics = complexity_analyzer.analyze_input_complexity(case['texts'])
        print(f"   Complexity score: {complexity_metrics.complexity_score:.3f}")
        
        # Simulate processing and get recommended batch size
        start_time = time.time()
        
        # Simulate some processing time based on text complexity
        simulated_processing_time = complexity_metrics.complexity_score * 200  # ms
        tokens_processed = sum(len(text.split()) for text in case['texts'])
        
        recommended_size = batch_manager.get_optimal_batch_size(
            processing_time_ms=simulated_processing_time,
            tokens_processed=tokens_processed,
            input_data=case['texts']
        )
        
        end_time = time.time()
        actual_time = (end_time - start_time) * 1000  # Convert to ms
        
        print(f"   Recommended batch size: {recommended_size}")
        print(f"   Simulated processing time: {simulated_processing_time:.1f}ms")
        print(f"   Tokens processed: {tokens_processed}")
        print(f"   Analysis time: {actual_time:.2f}ms")
    
    print("\n" + "-" * 60)
    print("FINAL STATUS REPORT")
    print("-" * 60)
    
    status = batch_manager.get_status_report()
    print(f"Current batch size: {status['current_batch_size']}")
    print(f"Average processing time: {status['average_processing_time_ms']:.2f}ms")
    print(f"Average throughput: {status['average_throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Text type distribution: {status['text_type_distribution']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


def demonstrate_performance_comparison():
    """Demonstrate performance differences with and without dynamic batching."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Create two batch managers - one with dynamic batching and one fixed
    dynamic_manager = get_dynamic_text_batch_manager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16
    )
    
    # Fixed batch size manager (we'll simulate this behavior)
    fixed_batch_size = 4
    
    print(f"Fixed batch size: {fixed_batch_size}")
    print(f"Dynamic batch manager initial size: {dynamic_manager.current_batch_size}")
    
    # Simulate processing of various text complexities
    text_complexities = [
        ("Simple", 0.2),   # Low complexity
        ("Moderate", 0.5), # Medium complexity  
        ("Complex", 0.8),  # High complexity
        ("Simple", 0.3),   # Low complexity again
    ]
    
    print("\nSimulating processing with different approaches:")
    print("-" * 60)
    
    for name, complexity in text_complexities:
        print(f"\nText type: {name} (complexity: {complexity:.2f})")
        
        # For fixed approach, we always use the same batch size
        fixed_processing_time = complexity * 100 + 50  # ms, varies with complexity
        print(f"  Fixed approach - Batch size: {fixed_batch_size}, Time: {fixed_processing_time:.1f}ms")
        
        # For dynamic approach, get recommended size based on complexity
        # Simulate processing with dynamic manager
        recommended_size = dynamic_manager.get_optimal_batch_size(
            processing_time_ms=complexity * 80 + 40,  # Simulated processing time
            tokens_processed=100,  # Fixed for comparison
            input_data=["sample text"] * int(4 * (1 - complexity))  # Vary input based on complexity
        )
        
        dynamic_processing_time = (complexity * 60 + 30) * (4 / recommended_size)  # Adjust for batch size
        print(f"  Dynamic approach - Batch size: {recommended_size}, Time: {dynamic_processing_time:.1f}ms")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_dynamic_text_batching()
    demonstrate_performance_comparison()