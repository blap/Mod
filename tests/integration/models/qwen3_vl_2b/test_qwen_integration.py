import sys
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)

sys.path.insert(0, r'C:\Users\Admin\Documents\GitHub\Mod')

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
# Create a minimal config for testing
config = Qwen3VL2BConfig()

# Disable some heavy optimizations for faster testing
config.model_path = "dummy_path"  # Avoid loading actual model
config.use_flash_attention_2 = False
config.use_cuda_kernels = False
config.enable_disk_offloading = False
config.enable_intelligent_pagination = False
config.enable_tensor_parallelism = False
config.use_multimodal_attention = False
config.enable_image_tokenization = True

print("Testing Qwen3-VL-2B model with image tokenization...")

# Mock the model loading to avoid downloading the actual model
with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained') as mock_model, \
     patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
     patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained') as mock_image_proc:

    # Set up mocks
    mock_model_instance = type('MockModel', (), {})()
    mock_model_instance.gradient_checkpointing_enable = lambda: None
    mock_model_instance.config = type('MockConfig', (), {})()
    mock_model_instance.config.hidden_size = 2048
    mock_model_instance.config.num_attention_heads = 16
    mock_model_instance.config.num_hidden_layers = 24
    mock_model.return_value = mock_model_instance

    mock_tokenizer_instance = type('MockTokenizer', (), {})()
    mock_tokenizer_instance.pad_token = None
    mock_tokenizer_instance.eos_token = '</s>'
    mock_tokenizer.return_value = mock_tokenizer_instance

    mock_image_proc_instance = type('MockImageProc', (), {})()
    mock_image_proc.return_value = mock_image_proc_instance

    # Create the model
    model = Qwen3VL2BModel(config)

    print("Model created successfully!")
    print(f"Image tokenizer initialized: {model._image_tokenizer is not None}")
    print(f"Model has image_tokenizer attribute: {hasattr(model._model, 'image_tokenizer')}")

    if model._image_tokenizer:
        print("Image tokenizer config:", model._image_tokenizer.config)

print("Test completed successfully!")