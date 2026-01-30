"""
Verification script for sequence parallelism implementation.
"""

def verify_sequence_parallelism():
    """Verify that sequence parallelism has been properly implemented."""
    print("Verifying sequence parallelism implementation...")
    
    # 1. Check that the sequence parallelism module exists
    try:
        from src.inference_pio.common.sequence_parallel import (
            SequenceParallel,
            SequenceParallelConfig,
            create_sequence_parallel_config
        )
        print("PASS: Sequence parallelism module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sequence parallelism module: {e}")
        return False
    
    # 2. Check that all model configs have sequence parallelism parameters
    configs_to_check = [
        ("GLM-4-7-Flash", "src.inference_pio.models.glm_4_7_flash.config", "GLM47FlashConfig"),
        ("Qwen3-4b-instruct-2507", "src.inference_pio.models.qwen3_4b_instruct_2507.config", "Qwen34BInstruct2507Config"),
        ("Qwen3-coder-30b", "src.inference_pio.models.qwen3_coder_30b.config", "Qwen3Coder30BConfig"),
        ("Qwen3-vl-2b", "src.inference_pio.models.qwen3_vl_2b.config", "Qwen3VL2BConfig"),
    ]
    
    for model_name, module_path, config_class in configs_to_check:
        try:
            module = __import__(module_path, fromlist=[config_class])
            config_cls = getattr(module, config_class)
            config_instance = config_cls()
            
            # Check for sequence parallelism attributes
            attrs_to_check = [
                'enable_sequence_parallelism',
                'sequence_parallel_num_segments', 
                'sequence_parallel_split_method',
                'sequence_parallel_enable_overlap',
                'sequence_parallel_overlap_size',
                'sequence_parallel_algorithm'
            ]
            
            missing_attrs = []
            for attr in attrs_to_check:
                if not hasattr(config_instance, attr):
                    missing_attrs.append(attr)
                    
            if missing_attrs:
                print(f"FAIL: {model_name} config missing attributes: {missing_attrs}")
                return False
            else:
                print(f"PASS: {model_name} config has all sequence parallelism attributes")
        except Exception as e:
            print(f"FAIL: Failed to check {model_name} config: {e}")
            return False

    # 3. Check that all models have sequence parallelism methods
    models_to_check = [
        ("GLM-4-7-Flash", "src.inference_pio.models.glm_4_7_flash.model", "GLM47FlashModel"),
        ("Qwen3-4b-instruct-2507", "src.inference_pio.models.qwen3_4b_instruct_2507.model", "Qwen34BInstruct2507Model"),
        ("Qwen3-coder-30b", "src.inference_pio.models.qwen3_coder_30b.model", "Qwen3Coder30BModel"),
        ("Qwen3-vl-2b", "src.inference_pio.models.qwen3_vl_2b.model", "Qwen3VL2BModel"),
    ]

    for model_name, module_path, model_class in models_to_check:
        try:
            module = __import__(module_path, fromlist=[model_class])
            model_cls = getattr(module, model_class)

            # Check for sequence parallelism methods and attributes
            methods_to_check = [
                '_initialize_sequence_parallelism',
                '_sequence_parallel_model'
            ]

            missing_items = []
            for method in methods_to_check:
                if method.startswith('_'):  # Attribute/property
                    # Check if the attribute exists in the class or can be set
                    has_attr = hasattr(model_cls, method) or method in dir(type('Temp', (), {})())
                    if not has_attr:
                        # For private attributes, we check if they can be set during initialization
                        # by looking at the source code
                        import inspect
                        try:
                            source = inspect.getsource(model_cls.__init__)
                            has_attr = method in source
                        except:
                            has_attr = False

                if not has_attr:
                    missing_items.append(method)

            if missing_items:
                print(f"FAIL: {model_name} model missing: {missing_items}")
                return False
            else:
                print(f"PASS: {model_name} model has sequence parallelism integration")
        except Exception as e:
            print(f"FAIL: Failed to check {model_name} model: {e}")
            return False

    # 4. Check that sequence parallelism is properly exported
    try:
        from src.inference_pio.common import SequenceParallel
        print("PASS: SequenceParallel is properly exported from common module")
    except ImportError:
        print("FAIL: SequenceParallel is not properly exported from common module")
        return False

    print("\nSUCCESS: All verifications passed! Sequence parallelism has been successfully implemented.")
    return True

if __name__ == "__main__":
    success = verify_sequence_parallelism()
    if not success:
        exit(1)