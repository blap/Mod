#!/usr/bin/env python3
"""
Script to update all model plugins to support sharding and streaming functionality.
"""

import os
import re
from pathlib import Path


def update_plugin_file(plugin_file_path: str):
    """Update a single plugin file to add sharding support."""
    with open(plugin_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add time import if not present
    if 'import time' not in content:
        content = content.replace(
            'import logging',
            'import logging\nimport time'
        )
    
    # Update the initialize method to include sharding initialization
    initialize_pattern = r'(def initialize\(self, \*\*kwargs\) -> bool:[\s\S]*?)(return True)'
    initialize_replacement = r'\1\n            # Initialize sharding if enabled in config\n            if getattr(self._config, \'enable_sharding\', False):\n                num_shards = getattr(self._config, \'num_shards\', 500)\n                storage_path = getattr(self._config, \'sharding_storage_path\', \'./shards/default\')\n                self.enable_sharding(num_shards=num_shards, storage_path=storage_path)\n                \n                # Shard the model\n                if self._model is not None:\n                    self.shard_model(self._model, num_shards=num_shards)\n\n\2'
    
    content = re.sub(initialize_pattern, initialize_replacement, content)
    
    # Update the infer method to use sharding if enabled
    infer_pattern = r'(def infer\(self, data: Any\) -> Any:[\s\S]*?)(try:)'
    infer_replacement = r'\1        # Use sharding if enabled\n        if self._sharding_enabled and self._model is not None:\n            return self._infer_with_sharding(data)\n\n\2'
    
    content = re.sub(infer_pattern, infer_replacement, content)
    
    # Add the sharding inference methods if they don't exist
    if '_infer_with_sharding' not in content:
        # Find the end of the infer method
        infer_end_pattern = r'(def infer\(self, data: Any\) -> Any:[\s\S]*?return generated_text[\s\n]*\n[\s\n]*\)[\s\n]*except'
        sharding_methods = r'''
    def _infer_with_sharding(self, data: str) -> str:
        """
        Perform inference using the sharding system.

        Args:
            data: Input text for inference

        Returns:
            Generated text
        """
        if not isinstance(data, str):
            raise ValueError("Model expects string input")

        # Handle empty input
        if not data.strip():
            logger.warning("Empty input provided, returning empty string")
            return ""

        try:
            # Tokenize input
            inputs = self._tokenizer(
                data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(self._config, 'max_input_length', 8192)
            )

            # Get input shape for shard selection
            input_shape = tuple(inputs['input_ids'].shape)

            # Prepare inference context
            context_id = f"infer_{int(time.time())}_{hash(data) % 10000}"
            loaded_shards = self.prepare_inference_context(context_id, input_shape, "forward")

            if not loaded_shards:
                logger.warning("No shards loaded for inference, falling back to regular inference")
                return self._fallback_infer(data)

            # Execute with shards
            input_tensor = inputs['input_ids']
            device = next(iter(self._sharder.loaded_shards.values())).parameters().__next__().device if self._sharder.loaded_shards else torch.device('cpu')
            input_tensor = input_tensor.to(device)

            output_tensor = self.execute_with_shards(context_id, input_tensor)

            # Clean up context
            self.cleanup_inference_context(context_id)

            # Convert output back to text (this is simplified - full generation with sharding is complex)
            generated_text = self._tokenizer.decode(
                output_tensor[0] if len(output_tensor.shape) > 1 else output_tensor,
                skip_special_tokens=True
            )

            return generated_text
        except Exception as e:
            logger.error(f"Error during sharded inference: {e}")
            # Fallback to regular inference
            return self._fallback_infer(data)

    def _fallback_infer(self, data: str) -> str:
        """
        Fallback inference method when sharding fails.

        Args:
            data: Input text for inference

        Returns:
            Generated text
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        # Tokenize input
        inputs = self._tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self._config, 'max_input_length', 8192)
        )

        # Move inputs to the same device as the model
        device = next(self._model.parameters()).device if self._model is not None else torch.device('cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use compiled model if available, otherwise use original model
        model_to_use = self.get_compiled_model()

        try:
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_length=min(len(inputs['input_ids'][0]) + getattr(self._config, 'max_new_tokens', 512), 
                                  getattr(self._config, 'max_input_length', 8192)),
                    pad_token_id=getattr(self._config, 'pad_token_id', 0),
                    do_sample=getattr(self._config, 'do_sample', True),
                    temperature=getattr(self._config, 'temperature', 0.7),
                    top_p=getattr(self._config, 'top_p', 0.9),
                    top_k=getattr(self._config, 'top_k', 50),
                    repetition_penalty=getattr(self._config, 'repetition_penalty', 1.0),
                    num_return_sequences=1,
                )

            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text
        except Exception as e:
            logger.error(f"Error during fallback inference: {e}")
            raise e
'''
        content = content.rstrip() + sharding_methods
    
    # Write the updated content back to the file
    with open(plugin_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {plugin_file_path}")


def main():
    """Main function to update all plugin files."""
    project_root = Path(__file__).parent
    plugin_dir = project_root / "src/inference_pio/models"
    
    # Find all plugin.py files in the model directories
    plugin_files = list(plugin_dir.rglob("plugin.py"))
    
    print(f"Found {len(plugin_files)} plugin files to update:")
    for plugin_file in plugin_files:
        print(f"  - {plugin_file}")
        update_plugin_file(str(plugin_file))
    
    print("\nAll plugin files have been updated with sharding support!")


if __name__ == "__main__":
    main()