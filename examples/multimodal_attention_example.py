"""
Example demonstrating the use of multimodal attention mechanisms in the Inference-PIO system.

This example shows how to configure and use the multimodal attention system
with different models in the Inference-PIO framework.
"""

import torch

from src.common.multimodal_attention import (
    AdaptiveMultimodalAttention,
    MultimodalCrossAttention,
    MultimodalFusionLayer,
)
from src.models.glm_4_7_flash.model import GLM47Config, GLM47Model
from src.models.qwen3_4b_instruct_2507.model import (
    Qwen34BInstruct2507Config,
    Qwen34BInstruct2507Model,
)
from src.models.qwen3_coder_30b.model import Qwen3Coder30BConfig, Qwen3Coder30BModel
from src.models.qwen3_vl_2b.model import Qwen3VL2BConfig, Qwen3VL2BModel


def create_sample_multimodal_inputs(batch_size=2, seq_len=10, d_model=512):
    """
    Create sample inputs for different modalities.
    """
    inputs = {
        "text": torch.randn(batch_size, seq_len, d_model),
        "image": torch.randn(batch_size, seq_len, d_model),
        "audio": torch.randn(batch_size, seq_len, d_model),
    }
    return inputs


def demonstrate_multimodal_attention():
    """
    Demonstrate the use of multimodal attention mechanisms.
    """
    print("Demonstrating Multimodal Attention Mechanisms")
    print("=" * 50)

    # Create sample inputs
    batch_size, seq_len, d_model = 2, 10, 512
    modalities = ["text", "image", "audio"]

    inputs = create_sample_multimodal_inputs(batch_size, seq_len, d_model)

    # Example 1: Basic Multimodal Cross Attention
    print("\n1. Basic Multimodal Cross Attention:")
    cross_attention = MultimodalCrossAttention(
        d_model=d_model, nhead=8, modalities=modalities
    )

    outputs, weights = cross_attention(
        queries=inputs, keys=inputs, values=inputs, need_weights=True
    )

    print(f"   Input shapes: {[inputs[m].shape for m in modalities]}")
    print(f"   Output shapes: {[outputs[m].shape for m in modalities]}")
    print(f"   Attention weight shapes: {[weights[m].shape for m in modalities]}")

    # Example 2: Multimodal Fusion Layer
    print("\n2. Multimodal Fusion Layer:")
    fusion_layer = MultimodalFusionLayer(
        d_model=d_model, nhead=8, modalities=modalities
    )

    fused_outputs = fusion_layer(inputs=inputs)
    print(f"   Input shapes: {[inputs[m].shape for m in modalities]}")
    print(f"   Fused output shapes: {[fused_outputs[m].shape for m in modalities]}")

    # Example 3: Adaptive Multimodal Attention
    print("\n3. Adaptive Multimodal Attention:")
    adaptive_attention = AdaptiveMultimodalAttention(
        d_model=d_model, nhead=8, modalities=modalities
    )

    adaptive_outputs, adaptive_weights = adaptive_attention(
        queries=inputs, keys=inputs, values=inputs, need_weights=True
    )

    print(f"   Input shapes: {[inputs[m].shape for m in modalities]}")
    print(
        f"   Adaptive output shapes: {[adaptive_outputs[m].shape for m in modalities]}"
    )


def demonstrate_model_integration():
    """
    Demonstrate how models can integrate multimodal attention.
    """
    print("\n\nDemonstrating Model Integration with Multimodal Attention")
    print("=" * 60)

    # Example configurations for enabling multimodal attention
    print("\n1. GLM-4-7 Model with Multimodal Attention:")
    glm_config = GLM47Config(model_path="fake_path")  # Will fall back to default
    # Add multimodal attributes dynamically
    glm_config.use_multimodal_attention = True
    glm_config.is_multimodal = True
    glm_config.modalities = ["text", "image", "audio"]
    glm_config.multimodal_dropout = 0.1
    print(
        f"   Configured with multimodal attention: {glm_config.use_multimodal_attention}"
    )
    print(f"   Modalities: {glm_config.modalities}")

    print("\n2. Qwen3-4B-Instruct-2507 Model with Multimodal Attention:")
    qwen4b_config = Qwen34BInstruct2507Config(
        model_path="fake_path"  # Will fall back to default
    )
    # Add multimodal attributes dynamically
    qwen4b_config.use_multimodal_attention = True
    qwen4b_config.is_multimodal = True
    qwen4b_config.modalities = ["text", "image", "audio"]
    qwen4b_config.multimodal_dropout = 0.1
    print(
        f"   Configured with multimodal attention: {qwen4b_config.use_multimodal_attention}"
    )
    print(f"   Modalities: {qwen4b_config.modalities}")

    print("\n3. Qwen3-Coder-30B Model with Multimodal Attention:")
    qwen_coder_config = Qwen3Coder30BConfig(
        model_path="fake_path"  # Will fall back to default
    )
    # Add multimodal attributes dynamically
    qwen_coder_config.use_multimodal_attention = True
    qwen_coder_config.is_multimodal = True
    qwen_coder_config.modalities = ["text", "code", "image"]
    qwen_coder_config.multimodal_dropout = 0.1
    print(
        f"   Configured with multimodal attention: {qwen_coder_config.use_multimodal_attention}"
    )
    print(f"   Modalities: {qwen_coder_config.modalities}")

    print("\n4. Qwen3-VL-2B Model with Enhanced Multimodal Attention:")
    qwen_vl_config = Qwen3VL2BConfig()
    # Add multimodal attributes dynamically
    qwen_vl_config.use_multimodal_attention = True
    qwen_vl_config.is_multimodal = True  # Default to True for VL models
    qwen_vl_config.modalities = ["text", "image", "audio"]
    qwen_vl_config.multimodal_dropout = 0.1
    print(
        f"   Configured with multimodal attention: {qwen_vl_config.use_multimodal_attention}"
    )
    print(f"   Modalities: {qwen_vl_config.modalities}")


def demonstrate_cross_modal_processing():
    """
    Demonstrate cross-modal processing with attention mechanisms.
    """
    print("\n\nDemonstrating Cross-Modal Processing")
    print("=" * 40)

    # Simulate processing where text queries image features
    batch_size, seq_len, d_model = 1, 5, 256
    modalities = ["text", "image"]

    # Create sample inputs
    text_input = torch.randn(batch_size, seq_len, d_model)  # Text tokens
    image_input = torch.randn(batch_size, seq_len, d_model)  # Image patches/features

    inputs = {"text": text_input, "image": image_input}

    print(f"Text input shape: {text_input.shape}")
    print(f"Image input shape: {image_input.shape}")

    # Create cross-attention where text queries image features
    cross_attention = MultimodalCrossAttention(
        d_model=d_model, nhead=8, modalities=modalities
    )

    # In this scenario, we want text to attend to image features
    outputs, weights = cross_attention(
        queries={"text": text_input},  # Text as queries
        keys=inputs,  # Both text and image as keys
        values=inputs,  # Both text and image as values
        need_weights=True,
    )

    print(f"\nAfter cross-attention:")
    print(f"Text output shape: {outputs['text'].shape}")
    print(f"Attention weights shape for text: {weights['text'].shape}")

    # Show how the text has now incorporated image information
    print(f"Cross-modal attention enables text to incorporate visual context")


if __name__ == "__main__":
    demonstrate_multimodal_attention()
    demonstrate_model_integration()
    demonstrate_cross_modal_processing()

    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("The multimodal attention system is ready for use in all models.")
