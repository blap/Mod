"""
Tests for general optimization profiles: performance and memory-efficient.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import shutil

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, 
    assert_is_none, assert_is_not_none, assert_in, assert_not_in, 
    assert_greater, assert_less, assert_is_instance, assert_raises, 
    run_tests
)


def test_performance_profile_creation():
    """Test creation of performance optimization profile."""
    from src.inference_pio.optimization.profiles import PerformanceProfile
    
    profile = PerformanceProfile(
        name="test_performance", 
        description="Test performance profile",
        optimization_level=3,
        enable_quantization=True,
        enable_pruning=True
    )
    
    assert_equal(profile.name, "test_performance", "Profile should have correct name")
    assert_equal(profile.description, "Test performance profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 3, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "Performance profile should enable quantization")
    assert_true(profile.enable_pruning, "Performance profile should enable pruning")


def test_memory_efficient_profile_creation():
    """Test creation of memory-efficient optimization profile."""
    from src.inference_pio.optimization.profiles import MemoryEfficientProfile
    
    profile = MemoryEfficientProfile(
        name="test_memory_efficient", 
        description="Test memory efficient profile",
        optimization_level=2,
        enable_quantization=True,
        enable_pruning=True,
        use_memory_mapping=True
    )
    
    assert_equal(profile.name, "test_memory_efficient", "Profile should have correct name")
    assert_equal(profile.description, "Test memory efficient profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 2, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "Memory efficient profile should enable quantization")
    assert_true(profile.enable_pruning, "Memory efficient profile should enable pruning")
    assert_true(profile.use_memory_mapping, "Memory efficient profile should use memory mapping")


def test_balanced_profile_creation():
    """Test creation of balanced optimization profile."""
    from src.inference_pio.optimization.profiles import BalancedProfile
    
    profile = BalancedProfile(
        name="test_balanced", 
        description="Test balanced profile",
        optimization_level=2,
        enable_quantization=True,
        enable_pruning=False,
        use_memory_mapping=False
    )
    
    assert_equal(profile.name, "test_balanced", "Profile should have correct name")
    assert_equal(profile.description, "Test balanced profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 2, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "Balanced profile should enable quantization")
    assert_false(profile.enable_pruning, "Balanced profile should not enable aggressive pruning")
    assert_false(profile.use_memory_mapping, "Balanced profile should not use aggressive memory mapping")


def test_profile_manager_registration():
    """Test that profile manager can register and retrieve optimization profiles."""
    from src.inference_pio.optimization.profiles import ProfileManager, PerformanceProfile
    
    # Create a temporary directory for profile storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the profile manager
        profile_manager = ProfileManager(profile_dir=test_dir)
        
        # Create a test profile
        profile = PerformanceProfile(
            name="test_profile",
            description="Test profile for registration"
        )
        
        # Register the profile
        result = profile_manager.register_profile("test_profile", profile)
        assert_true(result, "Profile registration should succeed")
        
        # Retrieve the profile
        retrieved = profile_manager.get_profile("test_profile")
        assert_is_not_none(retrieved, "Retrieved profile should not be None")
        assert_equal(retrieved.name, "test_profile", "Retrieved profile should have correct name")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_profile_application_to_config():
    """Test applying optimization profiles to configurations."""
    from src.inference_pio.optimization.profiles import PerformanceProfile
    from src.inference_pio.config.dynamic_config import GLM47DynamicConfig
    
    # Create a profile and a config
    profile = PerformanceProfile(
        name="apply_test",
        description="Test profile application",
        optimization_level=3,
        enable_quantization=True,
        enable_pruning=True
    )
    
    config = GLM47DynamicConfig(model_name="apply_test_config", temperature=0.7)
    
    # Apply the profile to the config
    optimized_config = profile.apply_to_config(config)
    
    # Check that the config has been modified appropriately
    assert_equal(optimized_config.model_name, "apply_test_config", "Config name should remain unchanged")
    # Note: The actual application logic depends on the implementation of apply_to_config


def test_profile_manager_list_profiles():
    """Test that profile manager can list optimization profiles."""
    from src.inference_pio.optimization.profiles import ProfileManager, PerformanceProfile, MemoryEfficientProfile
    
    # Create a temporary directory for profile storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the profile manager
        profile_manager = ProfileManager(profile_dir=test_dir)
        
        # Register multiple profiles
        perf_profile = PerformanceProfile(name="perf_test_1", description="Performance test 1")
        mem_profile = MemoryEfficientProfile(name="mem_test_1", description="Memory test 1")
        profile_manager.register_profile("perf_test_1", perf_profile)
        profile_manager.register_profile("mem_test_1", mem_profile)
        
        # List profiles
        profile_list = profile_manager.list_profiles()
        assert_greater(len(profile_list), 1, "Should have at least 2 profiles")
        assert_in("perf_test_1", profile_list, "Should contain performance profile")
        assert_in("mem_test_1", profile_list, "Should contain memory profile")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_profile_manager_delete_profile():
    """Test that profile manager can delete optimization profiles."""
    from src.inference_pio.optimization.profiles import ProfileManager, PerformanceProfile
    
    # Create a temporary directory for profile storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the profile manager
        profile_manager = ProfileManager(profile_dir=test_dir)
        
        # Register a profile
        profile = PerformanceProfile(name="delete_test", description="Delete test")
        profile_manager.register_profile("delete_test", profile)
        
        # Verify it exists
        retrieved = profile_manager.get_profile("delete_test")
        assert_is_not_none(retrieved, "Profile should exist before deletion")
        
        # Delete the profile
        delete_result = profile_manager.delete_profile("delete_test")
        assert_true(delete_result, "Profile deletion should succeed")
        
        # Verify it no longer exists
        deleted_retrieval = profile_manager.get_profile("delete_test")
        assert_is_none(deleted_retrieval, "Profile should not exist after deletion")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def run_tests():
    """Run all general optimization profile tests."""
    print("Running general optimization profile tests...")
    
    test_functions = [
        test_performance_profile_creation,
        test_memory_efficient_profile_creation,
        test_balanced_profile_creation,
        test_profile_manager_registration,
        test_profile_application_to_config,
        test_profile_manager_list_profiles,
        test_profile_manager_delete_profile
    ]
    
    all_passed = True
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All general optimization profile tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)