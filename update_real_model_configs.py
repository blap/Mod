#!/usr/bin/env python
"""
Configuration Utility for Real Model Tests and Benchmarks

This script updates configuration files to ensure proper H drive detection
and model path resolution for real model tests and benchmarks.
"""

import sys
import os
from pathlib import Path
import re


def update_config_files():
    """Update configuration files to ensure proper H drive detection."""
    print("Updating configuration files for H drive detection...")
    
    # Find all config files in the project
    config_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith("config.py") and "test" not in root:
                config_files.append(os.path.join(root, file))
    
    print(f"Found {len(config_files)} config files to update")
    
    # Pattern to find model path assignment in __post_init__
    model_path_pattern = r'(self\.model_path\s*=\s*get_default_model_path\(.*?\))'
    
    for config_file in config_files:
        print(f"Processing {config_file}...")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file already has H drive logic
        if 'H:/' in content or 'get_h_drive_base()' in content:
            print(f"  - Already has H drive logic, skipping")
            continue
        
        # Add H drive detection logic after model path assignment
        updated_content = re.sub(
            model_path_pattern,
            r'''        \1

        # Ensure the model path points to the H drive if available
        h_drive_path = self._get_h_drive_path()
        if h_drive_path and os.path.exists(h_drive_path):
            self.model_path = h_drive_path
            print(f"Using H drive model path: {h_drive_path}")
        elif not self.model_path or not os.path.exists(self.model_path):
            print(f"Model path not found: {self.model_path}, will attempt download or use cache")
''',
            content
        )
        
        # Add helper method if not already present
        if '_get_h_drive_path' not in updated_content:
            # Find the class definition
            class_pattern = r'(class\s+\w+.*?:.*?)(\n\s*def|\nclass|\nif __name__|\Z)'
            match = re.search(class_pattern, updated_content, re.DOTALL)
            
            if match:
                class_content = match.group(1)
                
                # Add the helper method before the next element
                helper_method = '''

    def _get_h_drive_path(self):
        """Get the H drive path for this model if available."""
        import os
        import platform
        
        # Determine the model-specific path on H drive
        model_name_clean = self.model_name.replace("_", "-").replace(" ", "")
        h_drive_paths = [
            f"H:/{model_name_clean}",
            f"H:/models/{model_name_clean}",
            f"H:/AI/models/{model_name_clean}",
        ]
        
        # Check platform-specific paths
        if platform.system() == 'Windows':
            for path in h_drive_paths:
                if os.path.exists(path):
                    return path
        else:
            # For Linux/WSL, common mount points
            mount_points = ['/mnt/h', '/media/h', '/drives/h']
            for mount_point in mount_points:
                for path in h_drive_paths:
                    alt_path = path.replace('H:/', f'{mount_point}/')
                    if os.path.exists(alt_path):
                        return alt_path
        
        return None
'''
                
                # Insert the helper method in the class
                updated_content = updated_content.replace(class_content, class_content + helper_method, 1)
        
        # Write the updated content back to the file
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"  - Updated {config_file}")


def update_plugin_files():
    """Update plugin files to ensure proper model loading with H drive priority."""
    print("\nUpdating plugin files for H drive model loading...")
    
    # Find all plugin files in the project
    plugin_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith("plugin.py") and "test" not in root:
                plugin_files.append(os.path.join(root, file))
    
    print(f"Found {len(plugin_files)} plugin files to update")
    
    for plugin_file in plugin_files:
        print(f"Processing {plugin_file}...")
        
        with open(plugin_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file already has H drive logic
        if 'H:/' in content or 'ModelLoader' in content:
            print(f"  - Already has H drive logic, skipping")
            continue
        
        # Add import for ModelLoader if not present
        if 'from ...core.model_loader' not in content and 'ModelLoader' not in content:
            # Add import after other imports
            import_section_end = content.find('\n', content.find('import')) + 1
            import_line = 'from ...core.model_loader import ModelLoader\n'
            content = content[:import_section_end] + import_line + content[import_section_end:]
        
        # Add model path resolution in initialize method
        if 'def initialize' in content:
            # Look for where model_path is used or set
            init_method_start = content.find('def initialize')
            init_method_end = content.find('\n    def ', init_method_start + 1)
            if init_method_end == -1:
                init_method_end = len(content)
            
            init_method = content[init_method_start:init_method_end]
            
            # Add model path resolution logic
            if 'ModelLoader.resolve_model_path' not in init_method:
                # Find where model loading occurs
                if 'model_path' in init_method:
                    # Add resolution logic before model loading
                    resolved_content = re.sub(
                        r'(.*?)(model_path.*?=.*?)(.*?)(\n.*?load.*?model)',
                        r'\1        # Resolve model path with H drive priority\n        resolved_path = ModelLoader.resolve_model_path(\n            self._config.model_name,\n            getattr(self._config, "hf_repo_id", None)\n        )\n        \2resolved_path\4',
                        init_method,
                        flags=re.DOTALL
                    )
                    
                    # Replace the initialize method in the full content
                    content = content[:init_method_start] + resolved_content + content[init_method_end:]
        
        # Write the updated content back to the file
        with open(plugin_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  - Updated {plugin_file}")


def update_test_files():
    """Update test files to ensure they work with real models and H drive detection."""
    print("\nUpdating test files for real model compatibility...")
    
    # Find all test files that import models
    test_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("test_") and file.endswith(".py") and "real" in file.lower():
                test_files.append(os.path.join(root, file))
    
    # Also include the main real test files we created
    test_files.extend([
        "updated_real_model_tests.py",
        "updated_real_model_benchmarks.py"
    ])
    
    print(f"Found {len(test_files)} test files to update")
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
            
        print(f"Processing {test_file}...")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add H drive detection import if not present
        if 'ModelLoader' not in content and 'h_drive' not in content.lower():
            # Add import after other imports
            import_pos = content.find('\n', content.find('import')) + 1
            import_line = 'from src.inference_pio.core.model_loader import ModelLoader\n'
            content = content[:import_pos] + import_line + content[import_pos:]
        
        # Write the updated content back to the file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  - Updated {test_file}")


def main():
    """Main function to update configurations for real model tests."""
    print("Starting configuration updates for real model tests and benchmarks...")
    
    update_config_files()
    update_plugin_files()
    update_test_files()
    
    print("\nConfiguration updates completed successfully!")
    print("Real model tests and benchmarks should now properly detect and use H drive models.")


if __name__ == "__main__":
    main()