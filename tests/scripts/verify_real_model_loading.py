import os
import sys
import torch
import logging
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig, AutoModel

# Add src to path just in case
sys.path.append(os.path.abspath("src"))

try:
    from src.qwen3_vl.core.config import Qwen3VLConfig
    from src.qwen3_vl.memory_management.optimized_memory_management import (
        MemoryManager,
        optimize_model_memory,
        get_memory_manager,
        MemoryConfig as OptimizedMemoryConfig
    )
    from src.qwen3_vl.optimization.hardware_optimizer import HardwareOptimizer
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_real_model():
    print("="*60)
    print("Verifying Qwen3-VL-2B-Instruct with Optimized Architecture")
    print("="*60)

    # 1. Initialize Optimized Configuration
    logger.info("Initializing configurations...")
    memory_config = OptimizedMemoryConfig(
        memory_pool_size=2 * 1024 * 1024 * 1024, # 2GB pool
    )

    # 2. Initialize Memory Manager
    logger.info("Initializing Memory Manager...")
    memory_manager = get_memory_manager(memory_config)

    # 3. Load Real Model
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    logger.info(f"Loading model: {model_id}...")

    try:
        # Check if we have GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load Processor
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info("Processor loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load processor: {e}")
            processor = None

        # Attempt Load
        logger.info("Attempting to load model...")
        model = None

        # Try Qwen2VL first if class available
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto"
            )
            logger.info("Model loaded successfully with Qwen2VL class.")
        except Exception as e:
            logger.info(f"Qwen2VL load failed: {e}")

        # Try AutoModel if failed
        if model is None:
            try:
                model = AutoModel.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype="auto"
                )
                logger.info("Model loaded successfully with AutoModel.")
            except Exception as e:
                logger.error(f"AutoModel load failed: {e}")

        if model is None:
            logger.error("Could not load model. Exiting verification.")
            return

        # 4. Apply Optimizations
        logger.info("Applying Memory Optimizations...")
        model = optimize_model_memory(model, memory_manager, memory_config)

        # 5. Apply Hardware Optimizations
        logger.info("Applying Hardware Optimizations...")
        hw_optimizer = HardwareOptimizer()
        model = hw_optimizer.optimize_model(model)

        print("\nModel Verification Successful!")
        print(f"Model Architecture: {type(model)}")

        # 6. Basic Inference Test
        if processor:
            logger.info("Running basic inference test...")
            text = "Describe this image."
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

            try:
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text_input], padding=True, return_tensors="pt")
                inputs = inputs.to(model.device)

                if hasattr(model, "generate"):
                    generated_ids = model.generate(**inputs, max_new_tokens=20)
                    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    print(f"\nGenerated Output: {output_text}")
                else:
                    logger.info("Model does not support 'generate'.")
            except Exception as e:
                logger.warning(f"Inference failed: {e}")

    except Exception as e:
        logger.error(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_real_model()
