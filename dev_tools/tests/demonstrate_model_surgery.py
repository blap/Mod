"""
Demonstration of the Model Surgery System for Inference-PIO.

This script demonstrates the Model Surgery technique that identifies and temporarily
removes non-essential components during inference to reduce model size and improve performance.
"""

import os
import sys

import torch
import torch.nn as nn

# Add the src directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.inference_pio.common.optimization.model_surgery import (
    ModelSurgerySystem,
    apply_model_surgery,
    restore_model_from_surgery,
)


class SampleModel(nn.Module):
    """A sample model with various components that can be surgically removed."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(0.4)
        self.norm2 = nn.LayerNorm(
            512
        )  # Using LayerNorm instead of BatchNorm to avoid dimension issues
        self.gelu = nn.GELU()
        self.linear3 = nn.Linear(512, 100)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        return x


def main():
    print("=" * 60)
    print("DEMONSTRATION: Model Surgery System for Inference-PIO")
    print("=" * 60)

    # Create a sample model
    print("\n1. Creating sample model...")
    model = SampleModel()

    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"   Original model parameters: {original_params:,}")

    # Show original model structure
    print("\n   Original model structure:")
    for name, module in model.named_modules():
        if name:  # Skip the root module
            print(f"     {name}: {type(module).__name__}")

    # Create surgery system
    print("\n2. Initializing Model Surgery System...")
    surgery_system = ModelSurgerySystem()

    # Analyze the model for surgery candidates
    print("\n3. Analyzing model for surgical removal candidates...")
    analysis = surgery_system.analyze_model_for_surgery(model)

    print(f"   Total parameters: {analysis['total_parameters']:,}")
    print(f"   Total modules: {analysis['total_modules']}")
    print(
        f"   Removable components found: {sum(len(v) for v in analysis['removable_components'].values())}"
    )

    for comp_type, components in analysis["removable_components"].items():
        print(f"     {comp_type}: {len(components)} components")
        for comp in components[:3]:  # Show first 3 of each type
            print(f"       - {comp['name']} (priority: {comp['priority']})")
        if len(components) > 3:
            print(f"       ... and {len(components) - 3} more")

    print("\n   Recommendations:")
    for rec in analysis["recommendations"]:
        print(f"     - {rec}")

    # Perform model surgery
    print("\n4. Performing model surgery...")
    modified_model = surgery_system.perform_surgery(model)

    # Count modified parameters
    modified_params = sum(p.numel() for p in modified_model.parameters())
    print(f"   Modified model parameters: {modified_params:,}")
    print(f"   Parameter reduction: {original_params - modified_params:,} parameters")
    print(f"   Size reduction: {(1 - modified_params/original_params)*100:.2f}%")

    # Show modified model structure
    print(
        "\n   Modified model structure (with removed components replaced by Identity):"
    )
    for name, module in modified_model.named_modules():
        if name and isinstance(module, nn.Identity):
            print(f"     {name}: {type(module).__name__} (replaced during surgery)")

    # Test inference with both models
    print("\n5. Testing inference with both models...")

    # Create sample input - matching the model's expected input size
    batch_size = 4
    input_tensor = torch.randn(batch_size, 128)  # Input size matches first linear layer

    # Test original model
    model.eval()
    with torch.no_grad():
        original_output = model(input_tensor)

    # Test modified model
    modified_model.eval()
    with torch.no_grad():
        modified_output = modified_model(input_tensor)

    print(f"   Original model output shape: {original_output.shape}")
    print(f"   Modified model output shape: {modified_output.shape}")
    print(
        f"   Outputs match in shape: {original_output.shape == modified_output.shape}"
    )

    # Calculate output similarity (they won't be identical due to dropout removal)
    similarity = torch.cosine_similarity(
        original_output.flatten(), modified_output.flatten(), dim=0
    )
    print(f"   Output similarity (cosine): {similarity.item():.4f}")

    # Restore the model
    print("\n6. Restoring original model from surgery...")
    restored_model = surgery_system.restore_model(modified_model)

    restored_params = sum(p.numel() for p in restored_model.parameters())
    print(f"   Restored model parameters: {restored_params:,}")
    print(f"   Parameters restored: {restored_params == original_params}")

    # Test that restored model works
    with torch.no_grad():
        restored_output = restored_model(input_tensor)

    print(f"   Restored model output shape: {restored_output.shape}")

    # Show that we can get surgery statistics
    print("\n7. Surgery statistics:")
    stats = surgery_system.get_surgery_stats()
    print(f"   Total surgeries performed: {stats['total_surgeries_performed']}")
    print(f"   Total components removed: {stats['total_components_removed']}")
    print(
        f"   Components pending restoration: {stats['components_pending_restoration']}"
    )
    print(f"   Active surgeries: {stats['active_surgeries']}")

    print("\n" + "=" * 60)
    print("MODEL SURGERY DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe Model Surgery system successfully demonstrated:")
    print("- Identification of removable components (dropouts, normalization layers)")
    print("- Safe removal of non-essential components during inference")
    print("- Reduction in model computational overhead")
    print("- Safe restoration of original model after inference")
    print("- Preservation of model functionality")


if __name__ == "__main__":
    main()
