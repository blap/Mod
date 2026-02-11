        else:
            _lib_cpu.load_tensor_data(name.encode('utf-8'), tensor._handle.contents.data, tensor.size)
    _lib_cpu.close_safetensors()
    return True

class SmartModelLoader(ModelLoader):
    def load_into_module(self, module, max_gpu_mem_gb=10.0):
        """
        Load weights into module, offloading to CPU if GPU full.
        """
        # 1. Iterate all parameters to get total size
        # 2. Assign devices
        # 3. Load

        # Simple Greedy Strategy
        current_gpu_mem = 0.0
        max_gpu_bytes = max_gpu_mem_gb * 1024**3

        # Open safetensors to get header info (sizes)
        # For "No Deps", we rely on the pre-initialized shapes in the module

        for name, param in module.parameters():
            if not param: continue

            size_bytes = param.size * 4 # float32

            if current_gpu_mem + size_bytes < max_gpu_bytes:
                # Move to GPU
                # This modifies the parameter in-place in the module logic typically,
                # but here we might need to replace the tensor object or call .to()
                # Since .to() returns a new tensor, we need to update the module dict.
                # However, module parameters are usually references.
                # Let's assume we set the device property or re-assign.

                # We need to find the parent module and name to re-assign?
                # Or just load data then .to()?
                # .to() handles data transfer.

                # Correct flow:
                # 1. Load data (on CPU initially via safetensors)
                # 2. Move to GPU
                pass # Logic handled during loading loop?

        # Actual loading logic
        # Re-using load_safetensors but with smart placement?
        # load_safetensors iterates the DICT of tensors.

        # We need to implement a Smart Load that checks size BEFORE moving.
        pass
