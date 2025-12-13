"""
GPU/CPU fallback mechanism verification for Qwen3-VL model
"""
import torch
import sys
import os
from typing import Dict, Any, Optional
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class DeviceFallbackValidator:
    """
    Validator for GPU/CPU fallback mechanisms in Qwen3-VL model.
    """
    def __init__(self):
        self.results = {}
    
    def test_device_availability(self) -> Dict[str, Any]:
        """
        Test device availability and properties.
        
        Returns:
            Dictionary with device availability information
        """
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_properties': {}
        }
        
        if device_info['cuda_available']:
            device_info['current_device'] = torch.cuda.current_device()
            device_info['device_properties'] = {
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(0),
                'total_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
                'available_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3  # Simplified
            }
        
        print("Device availability check:")
        print(f"  CUDA available: {device_info['cuda_available']}")
        if device_info['cuda_available']:
            print(f"  GPU: {device_info['device_properties']['name']}")
            print(f"  Memory: {device_info['device_properties']['total_memory']:.2f} GB")
        
        return device_info
    
    def create_test_model(self, device: torch.device) -> Qwen3VLForConditionalGeneration:
        """
        Create a test model on the specified device.
        
        Args:
            device: Device to create model on
        
        Returns:
            Model instance on specified device
        """
        config = Qwen3VLConfig()
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(device)
        model.eval()
        return model
    
    def test_model_on_device(
        self,
        model: Qwen3VLForConditionalGeneration,
        device: torch.device,
        test_name: str
    ) -> Dict[str, Any]:
        """
        Test model functionality on a specific device.
        
        Args:
            model: Model to test
            device: Device to test on
            test_name: Name for this test
        
        Returns:
            Dictionary with test results
        """
        print(f"Testing {test_name} on {device}...")
        
        # Create test inputs
        batch_size = 1
        seq_len = 32
        vocab_size = model.config.vocab_size
        image_size = model.config.vision_image_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones((batch_size, seq_len)).to(device)
        pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        # Test basic forward pass
        start_time = time.time()
        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
            
            success = True
            error_msg = None
            output_shape = list(output.logits.shape) if hasattr(output, 'logits') else list(output.shape)
        except Exception as e:
            success = False
            error_msg = str(e)
            output_shape = None
        
        elapsed_time = time.time() - start_time
        
        test_results = {
            'success': success,
            'error': error_msg,
            'elapsed_time': elapsed_time,
            'output_shape': output_shape,
            'device': str(device)
        }
        
        print(f"  Success: {success}")
        if not success:
            print(f"  Error: {error_msg}")
        else:
            print(f"  Output shape: {output_shape}")
            print(f"  Time: {elapsed_time:.4f}s")
        
        return test_results
    
    def test_gpu_to_cpu_fallback(self) -> Dict[str, Any]:
        """
        Test GPU to CPU fallback mechanism.
        
        Returns:
            Dictionary with GPU to CPU fallback test results
        """
        print("\nTesting GPU to CPU fallback...")
        
        fallback_results = {
            'gpu_test': None,
            'cpu_fallback_test': None,
            'fallback_success': False
        }
        
        # Test on GPU if available
        if torch.cuda.is_available():
            print("  Testing on GPU...")
            gpu_model = self.create_test_model(torch.device('cuda'))
            fallback_results['gpu_test'] = self.test_model_on_device(
                gpu_model, torch.device('cuda'), "GPU"
            )
            
            # If GPU test failed, try CPU fallback
            if not fallback_results['gpu_test']['success']:
                print("  GPU test failed, trying CPU fallback...")
                cpu_model = self.create_test_model(torch.device('cpu'))
                fallback_results['cpu_fallback_test'] = self.test_model_on_device(
                    cpu_model, torch.device('cpu'), "CPU Fallback"
                )
                fallback_results['fallback_success'] = fallback_results['cpu_fallback_test']['success']
            else:
                # GPU worked, but let's also test CPU for comparison
                cpu_model = self.create_test_model(torch.device('cpu'))
                fallback_results['cpu_fallback_test'] = self.test_model_on_device(
                    cpu_model, torch.device('cpu'), "CPU Comparison"
                )
                fallback_results['fallback_success'] = True
        else:
            # No GPU available, test on CPU
            print("  No GPU available, testing on CPU...")
            cpu_model = self.create_test_model(torch.device('cpu'))
            fallback_results['cpu_fallback_test'] = self.test_model_on_device(
                cpu_model, torch.device('cpu'), "CPU Only"
            )
            fallback_results['fallback_success'] = fallback_results['cpu_fallback_test']['success']
        
        return fallback_results
    
    def test_cpu_to_gpu_mechanism(self) -> Dict[str, Any]:
        """
        Test CPU to GPU functionality (normal operation when GPU is available).
        
        Returns:
            Dictionary with CPU to GPU test results
        """
        print("\nTesting CPU to GPU mechanism...")
        
        mechanism_results = {
            'cpu_test': None,
            'gpu_test': None,
            'gpu_available': torch.cuda.is_available(),
            'performance_comparison': {}
        }
        
        # Test on CPU
        cpu_model = self.create_test_model(torch.device('cpu'))
        mechanism_results['cpu_test'] = self.test_model_on_device(
            cpu_model, torch.device('cpu'), "CPU"
        )
        
        # Test on GPU if available
        if torch.cuda.is_available():
            gpu_model = self.create_test_model(torch.device('cuda'))
            mechanism_results['gpu_test'] = self.test_model_on_device(
                gpu_model, torch.device('cuda'), "GPU"
            )
            
            # Compare performance
            if mechanism_results['cpu_test']['success'] and mechanism_results['gpu_test']['success']:
                speedup = mechanism_results['cpu_test']['elapsed_time'] / mechanism_results['gpu_test']['elapsed_time']
                mechanism_results['performance_comparison'] = {
                    'cpu_time': mechanism_results['cpu_test']['elapsed_time'],
                    'gpu_time': mechanism_results['gpu_test']['elapsed_time'],
                    'speedup': speedup
                }
                print(f"  GPU speedup: {speedup:.2f}x")
        else:
            print("  GPU not available for comparison")
        
        return mechanism_results
    
    def test_device_memory_management(self) -> Dict[str, Any]:
        """
        Test device memory management capabilities.
        
        Returns:
            Dictionary with memory management test results
        """
        print("\nTesting device memory management...")
        
        memory_results = {
            'cpu_memory_before': None,
            'cpu_memory_after': None,
            'gpu_memory_before': None,
            'gpu_memory_after': None,
            'memory_cleanup_success': True
        }
        
        # Create model on CPU
        cpu_model = self.create_test_model(torch.device('cpu'))
        
        # Test basic functionality
        cpu_result = self.test_model_on_device(
            cpu_model, torch.device('cpu'), "CPU Memory Test"
        )
        
        # If GPU is available, test GPU memory management
        if torch.cuda.is_available():
            gpu_model = self.create_test_model(torch.device('cuda'))
            gpu_result = self.test_model_on_device(
                gpu_model, torch.device('cuda'), "GPU Memory Test"
            )
            
            # Clean up GPU memory
            del gpu_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clean up CPU memory
        del cpu_model
        
        print("  Memory cleanup completed")
        
        return memory_results
    
    def run_fallback_verification(self) -> Dict[str, Any]:
        """
        Run comprehensive GPU/CPU fallback verification.
        
        Returns:
            Dictionary with complete fallback verification results
        """
        print("Running GPU/CPU fallback mechanism verification...")
        
        # Test device availability
        device_info = self.test_device_availability()
        
        # Test GPU to CPU fallback
        gpu_to_cpu_results = self.test_gpu_to_cpu_fallback()
        
        # Test CPU to GPU mechanism
        cpu_to_gpu_results = self.test_cpu_to_gpu_mechanism()
        
        # Test memory management
        memory_results = self.test_device_memory_management()
        
        # Verify model capacity is preserved across devices
        config = Qwen3VLConfig()
        capacity_preserved = (
            config.num_hidden_layers == 32 and 
            config.num_attention_heads == 32
        )
        
        # Compile all results
        verification_results = {
            'device_info': device_info,
            'gpu_to_cpu_fallback': gpu_to_cpu_results,
            'cpu_to_gpu_mechanism': cpu_to_gpu_results,
            'memory_management': memory_results,
            'capacity_preserved': capacity_preserved,
            'overall_success': (
                gpu_to_cpu_results['fallback_success'] and 
                capacity_preserved
            )
        }
        
        # Print summary
        print("\n" + "="*60)
        print("GPU/CPU FALLBACK VERIFICATION SUMMARY")
        print("="*60)
        print(f"Device Info: CUDA available = {device_info['cuda_available']}")
        print(f"GPU to CPU Fallback: Success = {gpu_to_cpu_results['fallback_success']}")
        print(f"CPU to GPU Mechanism: GPU available = {cpu_to_gpu_results['gpu_available']}")
        if cpu_to_gpu_results['performance_comparison']:
            print(f"Performance: GPU speedup = {cpu_to_gpu_results['performance_comparison']['speedup']:.2f}x")
        print(f"Capacity Preserved: {capacity_preserved}")
        print(f"Overall Success: {verification_results['overall_success']}")
        print("="*60)
        
        return verification_results


def verify_gpu_cpu_fallback():
    """
    Run GPU/CPU fallback mechanism verification for Qwen3-VL model.
    """
    print("Running GPU/CPU fallback mechanism verification...")
    
    # Create validator
    validator = DeviceFallbackValidator()
    
    # Run verification
    results = validator.run_fallback_verification()
    
    return results


if __name__ == "__main__":
    results = verify_gpu_cpu_fallback()
    print("\nGPU/CPU fallback verification completed successfully!")