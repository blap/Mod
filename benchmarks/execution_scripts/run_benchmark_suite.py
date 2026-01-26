"""
Script to execute the comprehensive benchmark suite
"""

import subprocess
import sys
import os

def run_benchmark():
    """Run the comprehensive benchmark."""
    print("Running comprehensive benchmark suite...")
    
    # Run the benchmark
    try:
        result = subprocess.run([
            sys.executable, 
            "-m", 
            "final_comprehensive_benchmark"
        ], cwd=os.getcwd(), capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Benchmark timed out after 5 minutes")
    except Exception as e:
        print(f"Error running benchmark: {e}")

if __name__ == "__main__":
    run_benchmark()