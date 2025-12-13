"""
Memory profiling tools for Qwen3-VL model
"""
import torch
import psutil
import os
import sys
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class MemoryProfiler:
    """
    Comprehensive memory profiler for Qwen3-VL model.
    """
    def __init__(self):
        self.profiles = []
        self.cpu_memory_readings = []
        self.gpu_memory_readings = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        """
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU memory if available
        gpu_memory_mb = 0
        gpu_peak_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_peak_memory_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
        
        return {
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'gpu_peak_memory_mb': gpu_peak_memory_mb
        }
    
    def profile_model_memory(
        self,
        model: torch.nn.Module,
        input_data: Dict[str, torch.Tensor],
        label: str = "model"
    ) -> Dict[str, Any]:
        """
        Profile memory usage for a specific model operation.
        
        Args:
            model: The model to profile
            input_data: Input data for the model
            label: Label for this profiling session
        
        Returns:
            Dictionary with memory usage information
        """
        # Record initial memory
        initial_memory = self.get_memory_usage()
        
        # Run the model operation
        model.eval()
        with torch.no_grad():
            _ = model(**input_data)
        
        # Record memory after operation
        final_memory = self.get_memory_usage()
        
        # Calculate differences
        memory_diff = {
            'cpu_memory_diff_mb': final_memory['cpu_memory_diff_mb'] if 'cpu_memory_diff_mb' in final_memory 
                                  else final_memory['cpu_memory_mb'] - initial_memory['cpu_memory_mb'],
            'gpu_memory_diff_mb': final_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb'],
            'gpu_peak_memory_mb': final_memory['gpu_peak_memory_mb'],
            'initial_cpu_memory_mb': initial_memory['cpu_memory_mb'],
            'final_cpu_memory_mb': final_memory['cpu_memory_mb'],
            'initial_gpu_memory_mb': initial_memory['gpu_memory_mb'],
            'final_gpu_memory_mb': final_memory['gpu_memory_mb'],
            'label': label
        }
        
        # Store for later analysis
        self.profiles.append(memory_diff)
        
        return memory_diff
    
    def profile_model_components(
        self,
        model: Qwen3VLForConditionalGeneration,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Profile memory usage for different components of the model.
        
        Args:
            model: The Qwen3-VL model to profile
            input_ids: Text input IDs
            pixel_values: Image pixel values
        
        Returns:
            Dictionary with component-wise memory usage
        """
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)
        
        component_memory = {}
        
        # Profile vision encoder
        print("Profiling vision encoder...")
        with torch.no_grad():
            initial_memory = self.get_memory_usage()
            vision_features = model.vision_tower(pixel_values)
            vision_memory = self.get_memory_usage()
            component_memory['vision_encoder'] = {
                'cpu_memory_mb': vision_memory['cpu_memory_mb'] - initial_memory['cpu_memory_mb'],
                'gpu_memory_mb': vision_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb'],
                'output_size': list(vision_features.shape)
            }
        
        # Profile multimodal projector
        print("Profiling multimodal projector...")
        with torch.no_grad():
            initial_memory = self.get_memory_usage()
            projected_features = model.multi_modal_projector(vision_features)
            projector_memory = self.get_memory_usage()
            component_memory['multimodal_projector'] = {
                'cpu_memory_mb': projector_memory['cpu_memory_mb'] - vision_memory['cpu_memory_mb'],
                'gpu_memory_mb': projector_memory['gpu_memory_mb'] - vision_memory['gpu_memory_mb'],
                'output_size': list(projected_features.shape)
            }
        
        # Profile language model (text only)
        print("Profiling language model (text only)...")
        with torch.no_grad():
            initial_memory = self.get_memory_usage()
            text_output = model.language_model(input_ids=input_ids)
            text_memory = self.get_memory_usage()
            component_memory['language_model_text'] = {
                'cpu_memory_mb': text_memory['cpu_memory_mb'] - projector_memory['cpu_memory_mb'],
                'gpu_memory_mb': text_memory['gpu_memory_mb'] - projector_memory['gpu_memory_mb'],
                'output_size': list(text_output.shape)
            }
        
        # Profile full multimodal forward pass
        print("Profiling full multimodal forward pass...")
        combined_input = {
            'input_ids': input_ids,
            'pixel_values': pixel_values
        }
        full_memory = self.profile_model_memory(model, combined_input, "full_multimodal")
        component_memory['full_multimodal'] = full_memory
        
        return component_memory
    
    def profile_with_different_inputs(
        self,
        model: torch.nn.Module,
        base_input_ids: torch.Tensor,
        base_pixel_values: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Profile memory usage with different input sizes.
        
        Args:
            model: The model to profile
            base_input_ids: Base text input IDs
            base_pixel_values: Base image pixel values
        
        Returns:
            Dictionary with input size vs memory usage
        """
        device = next(model.parameters()).device
        
        # Different text lengths
        text_lengths = [16, 32, 64, 128, 256]
        text_memory_usage = []
        
        for length in text_lengths:
            print(f"Profiling with text length {length}...")
            input_ids = base_input_ids[:, :length].to(device)
            pixel_values = base_pixel_values.to(device)
            
            input_data = {
                'input_ids': input_ids,
                'pixel_values': pixel_values
            }
            
            memory_info = self.profile_model_memory(model, input_data, f"text_len_{length}")
            text_memory_usage.append({
                'text_length': length,
                'cpu_memory_mb': memory_info['cpu_memory_diff_mb'],
                'gpu_memory_mb': memory_info['gpu_memory_diff_mb']
            })
        
        # Different batch sizes
        batch_sizes = [1, 2, 4, 8]
        batch_memory_usage = []
        
        for batch_size in batch_sizes:
            print(f"Profiling with batch size {batch_size}...")
            try:
                input_ids = base_input_ids.repeat(batch_size, 1).to(device)
                pixel_values = base_pixel_values.repeat(batch_size, 1, 1, 1).to(device)
                
                input_data = {
                    'input_ids': input_ids,
                    'pixel_values': pixel_values
                }
                
                memory_info = self.profile_model_memory(model, input_data, f"batch_size_{batch_size}")
                batch_memory_usage.append({
                    'batch_size': batch_size,
                    'cpu_memory_mb': memory_info['cpu_memory_diff_mb'],
                    'gpu_memory_mb': memory_info['gpu_memory_diff_mb']
                })
            except RuntimeError as e:
                print(f"Batch size {batch_size} failed: {str(e)}")
                batch_memory_usage.append({
                    'batch_size': batch_size,
                    'cpu_memory_mb': None,
                    'gpu_memory_mb': None,
                    'error': str(e)
                })
        
        return {
            'text_length_memory': text_memory_usage,
            'batch_size_memory': batch_memory_usage
        }
    
    def generate_memory_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive memory usage report.
        
        Args:
            save_path: Optional path to save the report
        
        Returns:
            Memory usage report as a string
        """
        if not self.profiles:
            return "No profiles available."
        
        # Calculate statistics
        cpu_memory_diffs = [p['cpu_memory_diff_mb'] for p in self.profiles]
        gpu_memory_diffs = [p['gpu_memory_diff_mb'] for p in self.profiles]
        
        report = []
        report.append("="*60)
        report.append("QWEN3-VL MEMORY USAGE PROFILING REPORT")
        report.append("="*60)
        report.append(f"Total profiles: {len(self.profiles)}")
        report.append(f"CPU Memory - Avg: {np.mean(cpu_memory_diffs):.2f}MB, Max: {np.max(cpu_memory_diffs):.2f}MB, Min: {np.min(cpu_memory_diffs):.2f}MB")
        report.append(f"GPU Memory - Avg: {np.mean(gpu_memory_diffs):.2f}MB, Max: {np.max(gpu_memory_diffs):.2f}MB, Min: {np.min(gpu_memory_diffs):.2f}MB")
        
        # Detailed breakdown by label
        report.append("\nDETAILED BREAKDOWN:")
        for profile in self.profiles:
            report.append(f"  {profile['label']}: CPU={profile['cpu_memory_diff_mb']:.2f}MB, GPU={profile['gpu_memory_diff_mb']:.2f}MB")
        
        report.append("="*60)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
        
        return report_str


def run_memory_profiling():
    """
    Run comprehensive memory profiling for Qwen3-VL model.
    """
    print("Running memory profiling for Qwen3-VL model...")
    
    # Create model configuration
    config = Qwen3VLConfig()
    
    # Verify capacity is preserved
    assert config.num_hidden_layers == 32, f"Expected 32 layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"
    
    print(f"Configuration verified: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    
    # Initialize profiler
    profiler = MemoryProfiler()
    
    # Test on both CPU and GPU if available
    devices_to_test = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device('cuda'))
    
    all_results = {}
    
    for device in devices_to_test:
        print(f"\nProfiling on {device}...")
        model = model.to(device)
        model.eval()
        
        # Create test inputs
        batch_size = 1
        seq_len = 64
        vocab_size = config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(
            batch_size, 3, config.vision_image_size, config.vision_image_size
        )
        
        # Profile different model components
        print("Profiling model components...")
        component_results = profiler.profile_model_components(
            model, input_ids, pixel_values
        )
        
        # Profile with different input sizes
        print("Profiling with different input sizes...")
        input_results = profiler.profile_with_different_inputs(
            model, input_ids, pixel_values
        )
        
        all_results[str(device)] = {
            'components': component_results,
            'input_variations': input_results
        }
    
    # Generate and print report
    report = profiler.generate_memory_report()
    print(report)
    
    # Print component-specific results
    print("\nCOMPONENT MEMORY USAGE DETAILS:")
    for device, results in all_results.items():
        print(f"\n{device.upper()} COMPONENT RESULTS:")
        components = results['components']
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict) and 'cpu_memory_mb' in comp_data:
                print(f"  {comp_name}: CPU={comp_data['cpu_memory_mb']:.2f}MB, GPU={comp_data['gpu_memory_mb']:.2f}MB")
            else:
                print(f"  {comp_name}: {comp_data}")
    
    # Print input variation results
    print("\nINPUT VARIATION RESULTS:")
    for device, results in all_results.items():
        print(f"\n{device.upper()} INPUT VARIATIONS:")
        input_vars = results['input_variations']
        
        print("  Text Length Memory Usage:")
        for item in input_vars['text_length_memory']:
            print(f"    Length {item['text_length']}: CPU={item['cpu_memory_mb']:.2f}MB, GPU={item['gpu_memory_mb']:.2f}MB")
        
        print("  Batch Size Memory Usage:")
        for item in input_vars['batch_size_memory']:
            if 'error' in item:
                print(f"    Batch {item['batch_size']}: ERROR - {item['error']}")
            else:
                print(f"    Batch {item['batch_size']}: CPU={item['cpu_memory_mb']:.2f}MB, GPU={item['gpu_memory_mb']:.2f}MB")
    
    # Final summary
    print("\n" + "="*60)
    print("MEMORY PROFILING SUMMARY")
    print("="*60)
    for device, results in all_results.items():
        comp_memory = results['components']
        full_model_memory = comp_memory.get('full_multimodal', {})
        if full_model_memory:
            print(f"{device.upper()} - Full multimodal pass: CPU={full_model_memory.get('cpu_memory_diff_mb', 0):.2f}MB, GPU={full_model_memory.get('gpu_memory_diff_mb', 0):.2f}MB")
    print(f"Model capacity: {config.num_hidden_layers} layers, {config.num_attention_heads} heads (preserved: True)")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    results = run_memory_profiling()
    print("\nMemory profiling completed successfully!")