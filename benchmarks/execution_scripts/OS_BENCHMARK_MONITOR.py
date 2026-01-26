"""
OS Benchmark Monitoring Dashboard

This script creates a monitoring dashboard that tracks the execution status of OS benchmarks
and provides real-time updates every 5 minutes as benchmarks run.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import os
import sys
import threading
import psutil
import torch


class OSBenchmarkMonitor:
    def __init__(self):
        self.results_dir = Path("benchmark_results/os_benchmarks")
        self.monitoring_active = False
        self.monitoring_thread = None
        self.status_updates = []
        
    def start_monitoring_dashboard(self):
        """Start the monitoring dashboard."""
        print("OS BENCHMARK MONITORING DASHBOARD")
        print("="*60)
        print("Monitoring benchmark execution status...")
        print("Checking every 5 minutes for updates")
        print("="*60)
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        try:
            while self.monitoring_active:
                self.display_current_status()
                time.sleep(300)  # Wait 5 minutes
        except KeyboardInterrupt:
            print("\nStopping monitoring dashboard...")
            self.monitoring_active = False
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.monitoring_active:
            # Collect system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            gpu_stats = {}
            if torch.cuda.is_available():
                gpu_stats = {
                    "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                    "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                }
            
            status_update = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "available_memory_gb": round(memory.available / (1024**3), 2),
                "gpu_stats": gpu_stats
            }
            
            self.status_updates.append(status_update)
            
            time.sleep(300)  # Wait 5 minutes
    
    def display_current_status(self):
        """Display the current status of benchmark execution."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BENCHMARK EXECUTION STATUS")
        print("-" * 60)
        
        # Check for benchmark result files
        if self.results_dir.exists():
            # Look for recent result files
            json_files = list(self.results_dir.rglob("*.json"))
            csv_files = list(self.results_dir.rglob("*.csv"))
            
            print(f"Found {len(json_files)} JSON result files")
            print(f"Found {len(csv_files)} CSV result files")
            
            # Show most recent files
            if json_files:
                recent_json = max(json_files, key=lambda x: x.stat().st_mtime)
                mtime = datetime.fromtimestamp(recent_json.stat().st_mtime)
                print(f"Most recent JSON: {recent_json.name} (modified: {mtime.strftime('%H:%M:%S')})")
            
            if csv_files:
                recent_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                mtime = datetime.fromtimestamp(recent_csv.stat().st_mtime)
                print(f"Most recent CSV: {recent_csv.name} (modified: {mtime.strftime('%H:%M:%S')})")
        else:
            print("No benchmark results directory found yet")
        
        # Show latest system stats from our monitoring
        if self.status_updates:
            latest = self.status_updates[-1]
            print(f"\nSystem Stats (Last Update: {latest['timestamp']}):")
            print(f"  CPU Usage: {latest['cpu_percent']}%")
            print(f"  Memory Usage: {latest['memory_percent']}%")
            print(f"  Available Memory: {latest['available_memory_gb']} GB")
            if latest['gpu_stats']:
                print(f"  GPU Memory Allocated: {latest['gpu_stats']['gpu_memory_allocated_gb']} GB")
                print(f"  GPU Memory Reserved: {latest['gpu_stats']['gpu_memory_reserved_gb']} GB")
        else:
            print("\nSystem stats not available yet")
        
        # Check for ongoing processes
        benchmark_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if any('benchmark' in arg.lower() for arg in proc.info['cmdline'] or []):
                    benchmark_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        print(f"\nActive benchmark processes: {len(benchmark_processes)}")
        for proc in benchmark_processes[:3]:  # Show first 3
            print(f"  PID {proc['pid']}: {' '.join(proc['cmdline'][:3])}...")
        
        print("-" * 60)
    
    def generate_final_report(self):
        """Generate a final report of the monitoring."""
        if not self.status_updates:
            print("No monitoring data collected")
            return
        
        # Calculate averages
        avg_cpu = sum(u['cpu_percent'] for u in self.status_updates) / len(self.status_updates)
        avg_mem = sum(u['memory_percent'] for u in self.status_updates) / len(self.status_updates)
        
        # Find peaks
        peak_cpu = max(u['cpu_percent'] for u in self.status_updates)
        peak_mem = max(u['memory_percent'] for u in self.status_updates)
        
        print("\nMONITORING SUMMARY REPORT")
        print("="*60)
        print(f"Monitoring Period: {len(self.status_updates)} checks over {(self.status_updates[-1]['timestamp'] if self.status_updates else 'N/A')}")
        print(f"Average CPU Usage: {avg_cpu:.2f}%")
        print(f"Peak CPU Usage: {peak_cpu}%")
        print(f"Average Memory Usage: {avg_mem:.2f}%")
        print(f"Peak Memory Usage: {peak_mem}%")
        
        if self.status_updates[0]['gpu_stats']:
            avg_gpu_alloc = sum(u['gpu_stats']['gpu_memory_allocated_gb'] for u in self.status_updates if u['gpu_stats']) / len([u for u in self.status_updates if u['gpu_stats']])
            peak_gpu_alloc = max(u['gpu_stats']['gpu_memory_allocated_gb'] for u in self.status_updates if u['gpu_stats'])
            print(f"Average GPU Memory Allocated: {avg_gpu_alloc:.2f} GB")
            print(f"Peak GPU Memory Allocated: {peak_gpu_alloc:.2f} GB")


def main():
    """Main function for the monitoring dashboard."""
    monitor = OSBenchmarkMonitor()
    
    print("OS Benchmark Monitoring Dashboard")
    print("This dashboard monitors the execution status of OS benchmarks")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        monitor.start_monitoring_dashboard()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        monitor.generate_final_report()
    except Exception as e:
        print(f"Error in monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()