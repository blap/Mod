"""
Tests for model-specific optimization profiles.
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


def test_glm47_profile_creation():
    """Test creation of GLM-4.7 specific optimization profile."""
    from src.inference_pio.optimization.profiles import GLM47Profile
    
    profile = GLM47Profile(
        name="test_glm47", 
        description="Test GLM-4.7 profile",
        optimization_level=3,
        enable_quantization=True,
        enable_pruning=True,
        glm47_specific_optimization=True
    )
    
    assert_equal(profile.name, "test_glm47", "Profile should have correct name")
    assert_equal(profile.description, "Test GLM-4.7 profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 3, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "GLM47 profile should enable quantization")
    assert_true(profile.enable_pruning, "GLM47 profile should enable pruning")
    assert_true(profile.glm47_specific_optimization, "GLM47 profile should have specific optimization")


def test_qwen3_4b_profile_creation():
    """Test creation of Qwen3-4B specific optimization profile."""
    from src.inference_pio.optimization.profiles import Qwen34BProfile
    
    profile = Qwen34BProfile(
        name="test_qwen3_4b", 
        description="Test Qwen3-4B profile",
        optimization_level=2,
        enable_quantization=True,
        enable_pruning=False,
        qwen3_4b_specific_optimization=True
    )
    
    assert_equal(profile.name, "test_qwen3_4b", "Profile should have correct name")
    assert_equal(profile.description, "Test Qwen3-4B profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 2, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "Qwen3-4B profile should enable quantization")
    assert_false(profile.enable_pruning, "Qwen3-4B profile should not enable aggressive pruning")
    assert_true(profile.qwen3_4b_specific_optimization, "Qwen3-4B profile should have specific optimization")


def test_qwen3_coder_profile_creation():
    """Test creation of Qwen3-Coder specific optimization profile."""
    from src.inference_pio.optimization.profiles import Qwen3CoderProfile
    
    profile = Qwen3CoderProfile(
        name="test_qwen3_coder", 
        description="Test Qwen3-Coder profile",
        optimization_level=3,
        enable_quantization=True,
        enable_pruning=True,
        qwen3_coder_specific_optimization=True
    )
    
    assert_equal(profile.name, "test_qwen3_coder", "Profile should have correct name")
    assert_equal(profile.description, "Test Qwen3-Coder profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 3, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "Qwen3-Coder profile should enable quantization")
    assert_true(profile.enable_pruning, "Qwen3-Coder profile should enable pruning")
    assert_true(profile.qwen3_coder_specific_optimization, "Qwen3-Coder profile should have specific optimization")


def test_qwen3_vl_profile_creation():
    """Test creation of Qwen3-VL specific optimization profile."""
    from src.inference_pio.optimization.profiles import Qwen3VLProfile
    
    profile = Qwen3VLProfile(
        name="test_qwen3_vl", 
        description="Test Qwen3-VL profile",
        optimization_level=2,
        enable_quantization=True,
        enable_pruning=False,
        qwen3_vl_specific_optimization=True
    )
    
    assert_equal(profile.name, "test_qwen3_vl", "Profile should have correct name")
    assert_equal(profile.description, "Test Qwen3-VL profile", "Profile should have correct description")
    assert_equal(profile.optimization_level, 2, "Profile should have correct optimization level")
    assert_true(profile.enable_quantization, "Qwen3-VL profile should enable quantization")
    assert_false(profile.enable_pruning, "Qwen3-VL profile should not enable aggressive pruning")
    assert_true(profile.qwen3_vl_specific_optimization, "Qwen3-VL profile should have specific optimization")


def test_model_profile_compatibility():
    """Test compatibility between different model-specific profiles."""
    from src.inference_pio.optimization.profiles import (
        GLM47Profile, Qwen34BProfile, Qwen3CoderProfile, Qwen3VLProfile
    )
    
    # Create profiles for different models
    profiles = [
        GLM47Profile(name="glm47_compat", description="GLM47 compatibility test"),
        Qwen34BProfile(name="qwen3_4b_compat", description="Qwen3-4B compatibility test"),
        Qwen3CoderProfile(name="qwen3_coder_compat", description="Qwen3-Coder compatibility test"),
        Qwen3VLProfile(name="qwen3_vl_compat", description="Qwen3-VL compatibility test")
    ]
    
    # All should have common attributes
    for profile in profiles:
        assert_is_not_none(profile.name, "All profiles should have name")
        assert_is_not_none(profile.description, "All profiles should have description")
        assert_is_not_none(profile.optimization_level, "All profiles should have optimization level")


def test_model_profile_cloning():
    """Test cloning of model-specific optimization profiles."""
    from src.inference_pio.optimization.profiles import (
        GLM47Profile, Qwen34BProfile, Qwen3CoderProfile, Qwen3VLProfile
    )
    
    # Test cloning for each model-specific profile
    original_profiles = [
        GLM47Profile(name="clone_glm47", description="GLM47 clone test", glm47_specific_optimization=True),
        Qwen34BProfile(name="clone_qwen3_4b", description="Qwen3-4B clone test", qwen3_4b_specific_optimization=True),
        Qwen3CoderProfile(name="clone_qwen3_coder", description="Qwen3-Coder clone test", qwen3_coder_specific_optimization=True),
        Qwen3VLProfile(name="clone_qwen3_vl", description="Qwen3-VL clone test", qwen3_vl_specific_optimization=True)
    ]
    
    for original in original_profiles:
        cloned = original.clone(new_name=f"cloned_{original.name}")
        
        # Check that the clone has the new name
        assert_equal(cloned.name, f"cloned_{original.name}", "Cloned profile should have new name")
        
        # Check that other attributes are preserved
        assert_equal(cloned.description, original.description, "Description should be preserved in clone")
        assert_equal(cloned.optimization_level, original.optimization_level, "Optimization level should be preserved in clone")
        
        # Check that model-specific attributes are preserved
        if hasattr(original, 'glm47_specific_optimization'):
            assert_equal(cloned.glm47_specific_optimization, original.glm47_specific_optimization, "GLM47 specific optimization should be preserved")
        elif hasattr(original, 'qwen3_4b_specific_optimization'):
            assert_equal(cloned.qwen3_4b_specific_optimization, original.qwen3_4b_specific_optimization, "Qwen3-4b specific optimization should be preserved")
        elif hasattr(original, 'qwen3_coder_specific_optimization'):
            assert_equal(cloned.qwen3_coder_specific_optimization, original.qwen3_coder_specific_optimization, "Qwen3-Coder specific optimization should be preserved")
        elif hasattr(original, 'qwen3_vl_specific_optimization'):
            assert_equal(cloned.qwen3_vl_specific_optimization, original.qwen3_vl_specific_optimization, "Qwen3-VL specific optimization should be preserved")


def test_create_profile_from_template():
    """Test creating profiles from templates."""
    from src.inference_pio.optimization.profiles import GLM47Profile, ProfileTemplate
    
    # Create a template
    template = ProfileTemplate(
        name="template_base",
        description="Base template for profiles",
        optimization_level=2,
        enable_quantization=True,
        enable_pruning=False
    )
    
    # Create a profile from the template
    profile = GLM47Profile.from_template(
        template,
        name="profile_from_template",
        description="Profile created from template",
        glm47_specific_optimization=True
    )
    
    # Check that template properties are inherited
    assert_equal(profile.optimization_level, template.optimization_level, "Profile should inherit optimization level from template")
    assert_equal(profile.enable_quantization, template.enable_quantization, "Profile should inherit quantization setting from template")
    assert_equal(profile.enable_pruning, template.enable_pruning, "Profile should inherit pruning setting from template")
    
    # Check that profile-specific properties are set
    assert_equal(profile.name, "profile_from_template", "Profile should have its own name")
    assert_equal(profile.description, "Profile created from template", "Profile should have its own description")
    assert_true(profile.glm47_specific_optimization, "Profile should have model-specific property")


def test_profile_metadata():
    """Test profile metadata functionality."""
    from src.inference_pio.optimization.profiles import GLM47Profile
    
    profile = GLM47Profile(
        name="metadata_test",
        description="Profile for metadata testing",
        optimization_level=3,
        enable_quantization=True,
        enable_pruning=True,
        glm47_specific_optimization=True
    )
    
    # Check that metadata properties exist and are accessible
    assert_is_instance(profile.created_at, (str, type(None)), "Profile should have creation timestamp")
    assert_is_instance(profile.last_modified, (str, type(None)), "Profile should have last modified timestamp")
    assert_is_instance(profile.version, str, "Profile should have version")
    
    # Check that metadata can be updated
    original_version = profile.version
    profile.update_metadata({"notes": "Test note"})
    assert_equal(profile.version, original_version, "Version should not change with metadata update")


def run_tests():
    """Run all model-specific optimization profile tests."""
    print("Running model-specific optimization profile tests...")
    
    test_functions = [
        test_glm47_profile_creation,
        test_qwen3_4b_profile_creation,
        test_qwen3_coder_profile_creation,
        test_qwen3_vl_profile_creation,
        test_model_profile_compatibility,
        test_model_profile_cloning,
        test_create_profile_from_template,
        test_profile_metadata
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
        print("\n✓ All model-specific optimization profile tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)