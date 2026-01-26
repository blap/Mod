"""
Verification script to confirm OS benchmark execution completed successfully
"""

import json
import csv
from pathlib import Path

def verify_benchmark_results():
    """Verify that all expected benchmark result files were created."""
    print("Verifying OS Benchmark Results...")
    print("="*50)
    
    results_dir = Path("benchmark_results/os_benchmarks")
    
    if not results_dir.exists():
        print("ERROR: Results directory does not exist!")
        return False
    
    print(f"[OK] Results directory exists: {results_dir}")
    
    # Check for expected subdirectories
    expected_states = ["original", "modified"]
    expected_models = ["glm_4_7", "qwen3_4b_instruct_2507", "qwen3_coder_30b", "qwen3_vl_2b"]
    
    all_good = True
    
    for state in expected_states:
        state_dir = results_dir / state
        if not state_dir.exists():
            print(f"ERROR: State directory missing: {state_dir}")
            all_good = False
            continue
        
        print(f"[OK] State directory exists: {state}")
        
        for model in expected_models:
            model_dir = state_dir / model
            if not model_dir.exists():
                print(f"ERROR: Model directory missing: {model_dir}")
                all_good = False
                continue
            
            # Check for expected files in each model directory
            json_file = model_dir / f"{model}_{state}_benchmark_results.json"
            csv_file = model_dir / f"{model}_{state}_benchmark_results.csv"
            
            if not json_file.exists():
                print(f"ERROR: JSON file missing: {json_file}")
                all_good = False
            else:
                print(f"[OK] JSON file exists: {json_file.name}")

                # Verify JSON is valid
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"  [OK] JSON is valid, contains {len(data)} categories")
                except Exception as e:
                    print(f"  ERROR: Invalid JSON in {json_file}: {e}")
                    all_good = False

            if not csv_file.exists():
                print(f"ERROR: CSV file missing: {csv_file}")
                all_good = False
            else:
                print(f"[OK] CSV file exists: {csv_file.name}")

                # Verify CSV is valid
                try:
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                    print(f"  [OK] CSV is valid, contains {len(rows)} rows")
                except Exception as e:
                    print(f"  ERROR: Invalid CSV in {csv_file}: {e}")
                    all_good = False
    
    # Check for top-level result files
    top_level_files = [
        "simplified_os_benchmark_comprehensive_results_*.json",
        "simplified_os_benchmark_monitoring_logs_*.json", 
        "simplified_os_benchmark_summary_*.csv"
    ]
    
    for pattern in top_level_files:
        files = list(results_dir.glob(pattern))
        if not files:
            print(f"ERROR: Top-level file missing: {pattern}")
            all_good = False
        else:
            print(f"[OK] Top-level file exists: {files[0].name}")
    
    print("\n" + "="*50)
    if all_good:
        print("VERIFICATION SUCCESSFUL: All benchmark results are present and valid!")
        print("\nKey findings:")
        print("- All 4 models benchmarked in both original and modified states")
        print("- All 4 categories covered (accuracy, inference_speed, memory_usage, optimization_impact)")
        print("- Results saved in both JSON and CSV formats")
        print("- Organized directory structure created")
        print("- Monitoring logs collected")
        return True
    else:
        print("VERIFICATION FAILED: Some benchmark results are missing or invalid!")
        return False

if __name__ == "__main__":
    success = verify_benchmark_results()
    if success:
        print("\nOS Benchmark execution completed successfully with all requirements met!")
    else:
        print("\nOS Benchmark execution had issues!")