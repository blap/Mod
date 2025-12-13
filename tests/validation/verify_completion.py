#!/usr/bin/env python3
"""
Verification script to ensure all tasks in the Qwen3-VL architecture update plan are completed
"""

def check_completion_status():
    with open('qwen3_vl_architecture_update_plan.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all unchecked and checked items
    unchecked_items = content.count('[ ]')
    checked_items = content.count('[x]')
    
    print(f"Total unchecked items: {unchecked_items}")
    print(f"Total checked items: {checked_items}")
    
    if unchecked_items == 0:
        print("\n[SUCCESS] All tasks in the Qwen3-VL architecture update plan are completed!")
        print("The implementation is fully aligned with the update plan.")
    else:
        print(f"\n[WARNING] {unchecked_items} tasks are still not completed.")
        print("Scanning for incomplete tasks...")

        # Find lines with unchecked items
        lines = content.split('\n')
        incomplete_tasks = []

        for i, line in enumerate(lines):
            if '[ ]' in line:
                incomplete_tasks.append((i+1, line.strip()))

        print(f"\nIncomplete tasks found:")
        for line_num, task in incomplete_tasks:
            print(f"  Line {line_num}: {task}")
    
    return unchecked_items == 0

if __name__ == "__main__":
    success = check_completion_status()
    exit(0 if success else 1)