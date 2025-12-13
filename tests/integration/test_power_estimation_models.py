import unittest
import numpy as np
from power_estimation_models import (
    IntelI5_10210UPowerModel,
    NVidiaSM61PowerModel,
    PowerProfiler,
    estimate_cpu_power,
    estimate_gpu_power
)

class TestIntelI5_10210UPowerModel(unittest.TestCase):
    """Test cases for Intel i5-10210U power estimation model"""
    
    def setUp(self):
        self.model = IntelI5_10210UPowerModel()
    
    def test_idle_power_consumption(self):
        """Test power consumption at idle state"""
        power = self.model.estimate_power(utilization=0.0, frequency_ratio=0.1)
        self.assertGreaterEqual(power, 2.0)  # Minimum idle power
        self.assertLessEqual(power, 5.0)     # Maximum idle power
    
    def test_base_frequency_power(self):
        """Test power consumption at base frequency"""
        power = self.model.estimate_power(utilization=0.5, frequency_ratio=1.0)
        self.assertGreaterEqual(power, 8.0)   # Reasonable base power
        self.assertLessEqual(power, 15.0)     # Below TDP
    
    def test_boost_frequency_power(self):
        """Test power consumption at boost frequency"""
        power = self.model.estimate_power(utilization=0.8, frequency_ratio=1.75)  # 2.8 GHz (1.6*1.75)
        self.assertGreaterEqual(power, 12.0)  # Higher power at boost
        self.assertLessEqual(power, 25.0)     # Maximum under load
    
    def test_max_utilization_power(self):
        """Test power consumption at maximum utilization"""
        power = self.model.estimate_power(utilization=1.0, frequency_ratio=2.625)  # 4.2 GHz (1.6*2.625)
        self.assertGreaterEqual(power, 18.0)  # High utilization power
        self.assertLessEqual(power, 30.0)     # Maximum possible power
    
    def test_invalid_inputs(self):
        """Test model behavior with invalid inputs"""
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=-0.1, frequency_ratio=1.0)
        
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=1.1, frequency_ratio=1.0)
        
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=0.5, frequency_ratio=0.0)
        
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=0.5, frequency_ratio=5.0)


class TestNVidiaSM61PowerModel(unittest.TestCase):
    """Test cases for NVIDIA SM61 power estimation model"""
    
    def setUp(self):
        self.model = NVidiaSM61PowerModel()
    
    def test_idle_power_consumption(self):
        """Test power consumption at idle state"""
        power = self.model.estimate_power(utilization=0.0, memory_utilization=0.0)
        self.assertGreaterEqual(power, 1.0)  # Minimum idle power
        self.assertLessEqual(power, 3.0)     # Maximum idle power
    
    def test_low_utilization_power(self):
        """Test power consumption at low utilization"""
        power = self.model.estimate_power(utilization=0.2, memory_utilization=0.1)
        self.assertGreaterEqual(power, 3.0)  # Reasonable low power
        self.assertLessEqual(power, 8.0)     # Below average power
    
    def test_medium_utilization_power(self):
        """Test power consumption at medium utilization"""
        power = self.model.estimate_power(utilization=0.5, memory_utilization=0.4)
        self.assertGreaterEqual(power, 8.0)  # Medium power consumption
        self.assertLessEqual(power, 15.0)    # Below high power
    
    def test_high_utilization_power(self):
        """Test power consumption at high utilization"""
        power = self.model.estimate_power(utilization=0.9, memory_utilization=0.8)
        self.assertGreaterEqual(power, 15.0)  # High power consumption
        self.assertLessEqual(power, 25.0)     # Maximum for mobile GPU
    
    def test_max_utilization_power(self):
        """Test power consumption at maximum utilization"""
        power = self.model.estimate_power(utilization=1.0, memory_utilization=1.0)
        self.assertGreaterEqual(power, 20.0)  # Very high power consumption
        self.assertLessEqual(power, 25.0)     # Maximum for mobile GPU
    
    def test_invalid_inputs(self):
        """Test model behavior with invalid inputs"""
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=-0.1, memory_utilization=0.5)
        
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=1.1, memory_utilization=0.5)
        
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=0.5, memory_utilization=-0.1)
        
        with self.assertRaises(ValueError):
            self.model.estimate_power(utilization=0.5, memory_utilization=1.1)


class TestPowerProfiler(unittest.TestCase):
    """Test cases for power profiling utilities"""
    
    def setUp(self):
        self.profiler = PowerProfiler()
    
    def test_profile_cpu_power(self):
        """Test CPU power profiling functionality"""
        # Simulate a simple profiling session
        def dummy_cpu_work():
            # Simulate some CPU work
            result = 0
            for i in range(1000):
                result += i * i
            return result
        
        initial_power = self.profiler.get_current_cpu_power()
        result = dummy_cpu_work()
        final_power = self.profiler.get_current_cpu_power()
        
        # The result should be computed without errors
        self.assertIsNotNone(result)
        # Power values should be reasonable
        self.assertGreaterEqual(initial_power, 0)
        self.assertGreaterEqual(final_power, 0)
    
    def test_profile_gpu_power(self):
        """Test GPU power profiling functionality"""
        # For now, test that the method exists and returns reasonable values
        try:
            power = self.profiler.get_current_gpu_power()
            self.assertGreaterEqual(power, 0)
        except NotImplementedError:
            # If not implemented, that's OK for now
            pass


class TestEstimationFunctions(unittest.TestCase):
    """Test cases for high-level estimation functions"""
    
    def test_estimate_cpu_power(self):
        """Test high-level CPU power estimation"""
        power = estimate_cpu_power(0.7, 1.5)  # 70% utilization, 1.5x frequency ratio
        self.assertGreaterEqual(power, 0)
        self.assertLessEqual(power, 30)  # Should be within reasonable bounds
    
    def test_estimate_gpu_power(self):
        """Test high-level GPU power estimation"""
        power = estimate_gpu_power(0.6, 0.5)  # 60% utilization, 50% memory utilization
        self.assertGreaterEqual(power, 0)
        self.assertLessEqual(power, 25)  # Should be within reasonable bounds


class TestModelAccuracy(unittest.TestCase):
    """Test accuracy of power models against known values"""
    
    def test_cpu_model_accuracy(self):
        """Test CPU model against expected power ranges"""
        model = IntelI5_10210UPowerModel()
        
        # Test various utilization and frequency combinations
        test_cases = [
            # (utilization, freq_ratio, min_expected_power, max_expected_power)
            (0.0, 0.1, 2.0, 6.0),      # Idle
            (0.3, 1.0, 5.0, 12.0),     # Low load
            (0.6, 1.5, 10.0, 20.0),    # Medium load
            (0.9, 2.0, 15.0, 25.0),    # High load
            (1.0, 2.625, 18.0, 30.0),  # Max load at boost
        ]
        
        for utilization, freq_ratio, min_expected, max_expected in test_cases:
            power = model.estimate_power(utilization, freq_ratio)
            self.assertGreaterEqual(power, min_expected)
            self.assertLessEqual(power, max_expected)
    
    def test_gpu_model_accuracy(self):
        """Test GPU model against expected power ranges"""
        model = NVidiaSM61PowerModel()
        
        # Test various utilization combinations
        test_cases = [
            # (utilization, mem_util, min_expected_power, max_expected_power)
            (0.0, 0.0, 1.0, 4.0),    # Idle
            (0.2, 0.1, 3.0, 10.0),   # Low load
            (0.5, 0.4, 8.0, 17.0),   # Medium load
            (0.8, 0.7, 15.0, 24.0),  # High load
            (1.0, 1.0, 20.0, 25.0),  # Max load
        ]
        
        for utilization, mem_util, min_expected, max_expected in test_cases:
            power = model.estimate_power(utilization, mem_util)
            self.assertGreaterEqual(power, min_expected)
            self.assertLessEqual(power, max_expected)


if __name__ == '__main__':
    unittest.main()