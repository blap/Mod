"""
Verification Test for Optimized Models (Qwen3-VL, GLM-4.7-Flash)
"""

import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from inference_pio.core.engine.backend import Tensor, scaled_dot_product_attention
from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel, Qwen3VL2BConfig
from inference_pio.models.glm_4_7_flash.architecture import GLMForCausalLM
# GLM config is likely a SimpleNamespace or similar in tests?
class GLMConfig:
    def __init__(self, **kwargs):
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.num_hidden_layers = 2
        self.vocab_size = 100
        self.intermediate_size = 64
        self.layer_norm_eps = 1e-6
        self.max_position_embeddings = 128
        for k, v in kwargs.items(): setattr(self, k, v)

def test_backend_reshape_permute():
    """Verify backend reshape and permute."""
    # Create [1, 2, 3, 4] tensor (24 elements)
    t = Tensor([1, 2, 3, 4])
    data = [float(i) for i in range(24)]
    t.load(data)

    # Permute to [1, 3, 4, 2]
    # dims: [0, 2, 3, 1]
    p = t.permute([0, 2, 3, 1])
    assert p.shape == (1, 3, 4, 2)

    # Reshape to [1, 12, 2]
    r = p.reshape([1, 12, 2])
    assert r.shape == (1, 12, 2)
    assert r.size == 24

    print("Backend Reshape/Permute Passed")

def test_qwen3_vl_structure():
    """Verify Qwen3-VL-2B structure and forward pass."""
    config = Qwen3VL2BConfig()
    # Override for speed
    config.hidden_size = 32
    config.num_attention_heads = 4
    config.num_hidden_layers = 2
    config.vocab_size = 100
    config.vision_hidden_size = 16
    config.vision_num_attention_heads = 4
    config.vision_num_hidden_layers = 2
    config.vision_patch_size = 2 # Small patch
    config.vision_image_size = 4 # Small image (2x2 patches)
    config.vision_intermediate_size = 32
    config.intermediate_size = 64

    model = Qwen3VL2BModel(config)

    # Text input
    input_ids = Tensor([1, 5])
    input_ids.load([1, 2, 3, 4, 5])

    # Vision input [B, 3, H, W]
    # Image size 4x4
    pixel_values = Tensor([1, 3, 4, 4])
    pixel_values.fill(0.5)

    # Forward
    out = model(input_ids, pixel_values)

    # Expected output shape: [1, Seq_Text + Seq_Vis, Hidden]
    # Vis: (4/2)^2 = 4 patches.
    # Text: 5 tokens.
    # Total seq: 9.
    assert out.shape == (1, 9, 32)
    print("Qwen3-VL-2B Forward Passed")

def test_glm_4_7_flash_structure():
    """Verify GLM-4.7-Flash structure."""
    config = GLMConfig()
    model = GLMForCausalLM(config)

    input_ids = Tensor([1, 5])
    input_ids.load([1, 2, 3, 4, 5])

    logits, _ = model(input_ids)
    assert logits.shape == (1, 5, 100)
    print("GLM-4.7-Flash Forward Passed")

if __name__ == "__main__":
    test_backend_reshape_permute()
    test_qwen3_vl_structure()
    test_glm_4_7_flash_structure()
