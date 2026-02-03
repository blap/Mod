"""
Example usage of the Modular Testing Framework for Mod Project

This script demonstrates how to use the new modular testing framework.
"""

from testing_modules import TestOrchestrator, TestType
import sys
import os

# Add the src directory to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_example_tests():
    """Run example tests using the modular testing framework."""
    print("Modular Testing Framework - Example Usage")
    print("=" * 50)
    
    # Create the orchestrator
    orchestrator = TestOrchestrator()
    
    print("\n1. Running Functional Tests...")
    functional_result = orchestrator.run_tests_by_type(TestType.FUNCTIONAL, verbosity=1)
    
    print("\n2. Running Performance Tests...")
    performance_result = orchestrator.run_tests_by_type(TestType.PERFORMANCE, verbosity=1)
    
    print("\n3. Running Integration Tests...")
    integration_result = orchestrator.run_tests_by_type(TestType.INTEGRATION, verbosity=1)
    
    print("\n4. Running Regression Tests...")
    regression_result = orchestrator.run_tests_by_type(TestType.REGRESSION, verbosity=1)
    
    print("\n5. Running Unit Tests...")
    unit_result = orchestrator.run_tests_by_type(TestType.UNIT, verbosity=1)
    
    # Print summary
    orchestrator.print_test_summary()
    
    return orchestrator


def demonstrate_individual_module_usage():
    """Demonstrate how to use individual testing modules."""
    print("\n" + "=" * 50)
    print("Individual Module Usage Example")
    print("=" * 50)
    
    # Example of how to create a custom test using the functional testing module
    from testing_modules.functional_testing import ModelFunctionalTest
    from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin
    
    class ExampleFunctionalTest(ModelFunctionalTest):
        def get_model_plugin_class(self):
            return Qwen3_0_6B_Plugin
        
        def test_example_functionality(self):
            """An example test method."""
            # The model instance is available as self.model_instance
            # thanks to the base class setup
            self.assertIsNotNone(self.model_instance)
            print(f"  - Tested functionality for {self.model_instance.__class__.__name__}")
    
    # Run the example test
    import unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(ExampleFunctionalTest)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    print(f"  Example test ran with {len(result.failures)} failures and {len(result.errors)} errors")


if __name__ == "__main__":
    # Run example tests
    orchestrator = run_example_tests()
    
    # Demonstrate individual module usage
    demonstrate_individual_module_usage()
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("The modular testing framework is ready for use.")
    print("=" * 50)