"""
Unimodal Model Surgery System for Inference-PIO

This module extends the basic Model Surgery system to handle unimodal models,
specifically focusing on text-based models like GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b.
It provides specialized techniques for pruning and modifying components that handle
language-specific processing while maintaining integrity of textual information.
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .model_surgery import ComponentType, ModelSurgerySystem, SurgicalComponent

logger = logging.getLogger(__name__)


class UnimodalComponentType(Enum):
    """Extended types of unimodal model components that can be surgically modified."""

    # Standard types inherited from base system
    NORMALIZATION_LAYER = "normalization_layer"
    DROPOUT_LAYER = "dropout_layer"
    ACTIVATION_LAYER = "activation_layer"
    ATTENTION_MASK = "attention_mask"
    UNUSED_PARAMETER = "unused_parameter"
    REDUNDANT_CONNECTION = "redundant_connection"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"

    # Language-specific components
    EMBEDDING_LAYER = "embedding_layer"
    POSITIONAL_ENCODING = "positional_encoding"
    TOKEN_TYPE_EMBEDDING = "token_type_embedding"
    LANGUAGE_TRANSFORMER_BLOCK = "language_transformer_block"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    FEED_FORWARD_NETWORK = "feed_forward_network"
    LANGUAGE_ADAPTER = "language_adapter"

    # Text-specific components
    TEXT_EMBEDDING_LAYER = "text_embedding_layer"
    TEXT_ATTENTION_HEAD = "text_attention_head"
    TEXT_MLP_BLOCK = "text_mlp_block"
    TEXT_NORMALIZATION = "text_normalization"
    TEXT_ACTIVATION = "text_activation"


@dataclass
class UnimodalSurgicalComponent(SurgicalComponent):
    """Extends SurgicalComponent with unimodal-specific properties."""

    language_specificity: float = 1.0  # How specific to language processing (0.0-1.0)
    semantic_importance: float = 1.0  # Importance for semantic understanding (0.0-1.0)
    computational_overhead: float = 0.5  # Computational cost (0.0-1.0)


class UnimodalModelSurgerySystem(ModelSurgerySystem):
    """
    Advanced Model Surgery System specifically designed for unimodal text models.
    Focuses on maintaining integrity of textual information while optimizing performance.
    """

    def __init__(self):
        super().__init__()
        self.unimodal_components: Dict[str, UnimodalSurgicalComponent] = {}
        self.language_dependencies: Dict[str, Set[str]] = defaultdict(set)

    def identify_removable_components(
        self, model: nn.Module
    ) -> List[UnimodalSurgicalComponent]:
        """
        Identify unimodal components that can be safely removed during inference.

        Args:
            model: The neural network model to analyze

        Returns:
            List of removable unimodal components with their details
        """
        removable_components = []

        # Walk through all modules in the model
        for name, module in model.named_modules():
            component_type = self._classify_unimodal_module(module)

            if component_type:
                # Determine if this component can be safely removed
                can_remove, reason, priority = self._can_safely_remove_unimodal(
                    module, component_type, name
                )

                if can_remove:
                    # Analyze unimodal properties
                    language_specificity = self._analyze_language_specificity(
                        module, name
                    )
                    semantic_importance = self._analyze_semantic_importance(
                        module, name
                    )
                    computational_overhead = self._analyze_computational_overhead(
                        module
                    )

                    surgical_component = UnimodalSurgicalComponent(
                        name=name,
                        module=module,
                        type=component_type,
                        original_state=None,  # Will be set during removal
                        is_removed=False,
                        removal_reason=reason,
                        priority=priority,
                        language_specificity=language_specificity,
                        semantic_importance=semantic_importance,
                        computational_overhead=computational_overhead,
                    )
                    removable_components.append(surgical_component)

        # Sort by priority (lower numbers = higher priority for removal)
        # But prioritize low semantic importance over high priority
        removable_components.sort(key=lambda x: (x.priority, x.semantic_importance))

        return removable_components

    def _classify_unimodal_module(
        self, module: nn.Module
    ) -> Optional[UnimodalComponentType]:
        """
        Classify a module to determine its unimodal type for surgical purposes.

        Args:
            module: The module to classify

        Returns:
            UnimodalComponentType if the module is of interest, None otherwise
        """
        # First, try the base classification
        base_type = self._classify_module(module)
        if base_type:
            # Map base type to unimodal type
            try:
                return UnimodalComponentType(base_type.value)
            except ValueError:
                # If base type doesn't exist in unimodal enum, continue to unimodal-specific checks
                # Placeholder for actual model surgery implementation
                # This would contain the actual surgery algorithm
                logger.warning(f"Model surgery not implemented for layer type: {type(layer)}")
                return layer

        # Check for unimodal-specific components
        module_class_name = module.__class__.__name__.lower()

        # Language-specific components
        if "embedding" in module_class_name and "word" in module_class_name:
            return UnimodalComponentType.EMBEDDING_LAYER
        elif "positional" in module_class_name or "position" in module_class_name:
            return UnimodalComponentType.POSITIONAL_ENCODING
        elif "token_type" in module_class_name or "segment" in module_class_name:
            return UnimodalComponentType.TOKEN_TYPE_EMBEDDING
        elif "transformer" in module_class_name and "decoder" in module_class_name:
            return UnimodalComponentType.LANGUAGE_TRANSFORMER_BLOCK
        elif "attention" in module_class_name and "multi" in module_class_name:
            return UnimodalComponentType.MULTI_HEAD_ATTENTION
        elif "feedforward" in module_class_name or "mlp" in module_class_name:
            return UnimodalComponentType.FEED_FORWARD_NETWORK
        elif "adapter" in module_class_name and "language" in module_class_name:
            return UnimodalComponentType.LANGUAGE_ADAPTER

        # Text-specific components
        if "text" in module_class_name and "embedding" in module_class_name:
            return UnimodalComponentType.TEXT_EMBEDDING_LAYER
        elif "text" in module_class_name and "attention" in module_class_name:
            return UnimodalComponentType.TEXT_ATTENTION_HEAD
        elif "text" in module_class_name and "mlp" in module_class_name:
            return UnimodalComponentType.TEXT_MLP_BLOCK
        elif "text" in module_class_name and (
            "norm" in module_class_name or "layernorm" in module_class_name
        ):
            return UnimodalComponentType.TEXT_NORMALIZATION
        elif "text" in module_class_name and "relu" in module_class_name:
            return UnimodalComponentType.TEXT_ACTIVATION

        return None

    def _can_safely_remove_unimodal(
        self, module: nn.Module, component_type: UnimodalComponentType, name: str
    ) -> Tuple[bool, str, int]:
        """
        Determine if a unimodal component can be safely removed during inference.

        Args:
            module: The module to evaluate
            component_type: The type of unimodal component
            name: The name/path of the module

        Returns:
            Tuple of (can_remove, reason, priority)
        """
        # Check if the component type has a base equivalent
        try:
            base_component_type = ComponentType(component_type.value)
            # Use base logic first
            base_can_remove, base_reason, base_priority = self._can_safely_remove(
                module, base_component_type, name
            )

            if base_can_remove:
                return base_can_remove, base_reason, base_priority
        except ValueError:
            # Component type doesn't exist in base enum, continue with unimodal-specific logic
            # Placeholder for actual model surgery implementation
            # This would contain the actual surgery algorithm
            logger.warning(f"Surgery not implemented for layer: {layer_name}")
            return layer

        # Special handling for unimodal components
        if component_type == UnimodalComponentType.EMBEDDING_LAYER:
            # Embedding layers are usually critical for language models
            return (
                False,
                "Embedding layers are essential for language understanding",
                100,
            )

        elif component_type == UnimodalComponentType.POSITIONAL_ENCODING:
            # Positional encodings can sometimes be simplified
            return True, "Positional encoding can be simplified during inference", 5

        elif component_type == UnimodalComponentType.LANGUAGE_TRANSFORMER_BLOCK:
            # Language transformer blocks can sometimes be pruned if not critical
            return (
                True,
                "Language transformer block can be simplified during inference",
                8,
            )

        elif component_type == UnimodalComponentType.MULTI_HEAD_ATTENTION:
            # Multi-head attention can sometimes be simplified
            return True, "Multi-head attention can be simplified during inference", 7

        elif component_type == UnimodalComponentType.FEED_FORWARD_NETWORK:
            # Feed-forward networks can sometimes be simplified
            return True, "Feed-forward network can be simplified during inference", 7

        elif component_type == UnimodalComponentType.LANGUAGE_ADAPTER:
            # Language adapters can sometimes be simplified based on input language
            return True, "Language adapter can be simplified during inference", 6

        elif component_type in [
            UnimodalComponentType.TEXT_EMBEDDING_LAYER,
            UnimodalComponentType.TEXT_ATTENTION_HEAD,
            UnimodalComponentType.TEXT_MLP_BLOCK,
        ]:
            # Text-specific components - analyze based on their importance
            if "auxiliary" in name.lower() or "extra" in name.lower():
                return (
                    True,
                    f"{component_type.value} appears auxiliary and can be simplified",
                    6,
                )
            else:
                return (
                    False,
                    f"{component_type.value} is likely essential for text processing",
                    90,
                )

        # Default: don't remove
        return False, "Component is essential for unimodal text inference", 100

    def _analyze_language_specificity(self, module: nn.Module, name: str) -> float:
        """
        Analyze how specific a module is to language processing.

        Args:
            module: The module to analyze
            name: The name/path of the module

        Returns:
            Specificity score (0.0-1.0)
        """
        specificity = 0.5  # Base assumption

        name_lower = name.lower()

        # Increase specificity for language-specific components
        if any(
            keyword in name_lower
            for keyword in ["text", "language", "word", "token", "sentence", "phrase"]
        ):
            specificity += 0.3

        # Decrease for general components
        if any(keyword in name_lower for keyword in ["general", "universal", "shared"]):
            specificity -= 0.2

        return max(0.0, min(1.0, specificity))

    def _analyze_semantic_importance(self, module: nn.Module, name: str) -> float:
        """
        Analyze the semantic importance of a module for language understanding.

        Args:
            module: The module to analyze
            name: The name/path of the module

        Returns:
            Importance score (0.0-1.0)
        """
        importance = 0.5  # Base assumption

        name_lower = name.lower()

        # Increase importance for critical components
        if any(
            keyword in name_lower
            for keyword in ["encoder", "decoder", "attention", "transformer"]
        ):
            importance += 0.3
        elif any(
            keyword in name_lower
            for keyword in ["embedding", "vocab", "classification"]
        ):
            importance += 0.2

        # Decrease for auxiliary components
        if any(
            keyword in name_lower for keyword in ["aux", "auxiliary", "extra", "debug"]
        ):
            importance -= 0.3

        return max(0.0, min(1.0, importance))

    def _analyze_computational_overhead(self, module: nn.Module) -> float:
        """
        Analyze the computational overhead of a module.

        Args:
            module: The module to analyze

        Returns:
            Overhead score (0.0-1.0)
        """
        # This is a simplified implementation
        # In a real implementation, this would analyze the module's architecture
        # to determine its computational complexity
        return 0.5

    def perform_unimodal_surgery(
        self,
        model: nn.Module,
        components_to_remove: Optional[List[str]] = None,
        preserve_components: Optional[List[str]] = None,
        preserve_semantic_importance_threshold: float = 0.7,
    ) -> nn.Module:
        """
        Perform unimodal model surgery by removing specified components.

        Args:
            model: The model to perform surgery on
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable
            preserve_semantic_importance_threshold: Don't remove components with semantic importance above this threshold

        Returns:
            Modified model with components removed
        """
        # Identify components to remove if not specified
        if components_to_remove is None:
            all_removable = self.identify_removable_components(model)

            # Filter based on semantic importance threshold
            components_to_remove = [
                comp.name
                for comp in all_removable
                if comp.semantic_importance <= preserve_semantic_importance_threshold
            ]

        # Apply preservation filters
        if preserve_components:
            components_to_remove = [
                name for name in components_to_remove if name not in preserve_components
            ]

        # Perform the actual surgery using the base method
        return self.perform_surgery(model, components_to_remove, preserve_components)

    def analyze_unimodal_model_for_surgery(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze a unimodal model to provide recommendations for surgery.

        Args:
            model: The model to analyze

        Returns:
            Dictionary containing unimodal analysis results
        """
        analysis = {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "total_modules": 0,
            "removable_components": {},
            "potential_savings": 0,
            "recommendations": [],
            "language_specificity_distribution": defaultdict(int),
            "semantic_importance_analysis": {},
        }

        # Count total modules
        for name, module in model.named_modules():
            analysis["total_modules"] += 1
            # Count by language specificity
            name_lower = name.lower()
            if any(
                keyword in name_lower
                for keyword in ["text", "language", "word", "token"]
            ):
                analysis["language_specificity_distribution"]["high"] += 1
            elif any(keyword in name_lower for keyword in ["general", "shared"]):
                analysis["language_specificity_distribution"]["low"] += 1
            else:
                analysis["language_specificity_distribution"]["medium"] += 1

        # Identify removable components
        removable = self.identify_removable_components(model)

        for comp in removable:
            comp_type = comp.type.value
            if comp_type not in analysis["removable_components"]:
                analysis["removable_components"][comp_type] = []

            analysis["removable_components"][comp_type].append(
                {
                    "name": comp.name,
                    "reason": comp.removal_reason,
                    "priority": comp.priority,
                    "language_specificity": comp.language_specificity,
                    "semantic_importance": comp.semantic_importance,
                    "computational_overhead": comp.computational_overhead,
                }
            )

        # Generate recommendations
        if removable:
            analysis["recommendations"].append(
                f"Found {len(removable)} potentially removable unimodal components that could reduce model size"
            )

            # Count by type
            type_counts = defaultdict(int)
            for comp in removable:
                type_counts[comp.type.value] += 1

            for comp_type, count in type_counts.items():
                analysis["recommendations"].append(f"- {count} {comp_type} components")

            # Language-specific recommendations
            for lang_spec, count in analysis[
                "language_specificity_distribution"
            ].items():
                analysis["recommendations"].append(
                    f"- {count} {lang_spec}-specific components"
                )
        else:
            analysis["recommendations"].append(
                "No obvious candidates for surgical removal found"
            )

        return analysis


# Global instance for convenience
_unimodal_model_surgery_system = UnimodalModelSurgerySystem()


def get_unimodal_model_surgery_system() -> UnimodalModelSurgerySystem:
    """
    Get the global unimodal model surgery system instance.

    Returns:
        UnimodalModelSurgerySystem instance
    """
    return _unimodal_model_surgery_system


def apply_unimodal_model_surgery(
    model: nn.Module,
    components_to_remove: Optional[List[str]] = None,
    preserve_components: Optional[List[str]] = None,
    preserve_semantic_importance_threshold: float = 0.7,
) -> nn.Module:
    """
    Convenience function to apply unimodal model surgery to a model.

    Args:
        model: The model to perform surgery on
        components_to_remove: Specific components to remove (None means auto-detect)
        preserve_components: Components to preserve even if they're removable
        preserve_semantic_importance_threshold: Don't remove components with semantic importance above this threshold

    Returns:
        Modified model with surgery applied
    """
    surgery_system = get_unimodal_model_surgery_system()
    return surgery_system.perform_unimodal_surgery(
        model,
        components_to_remove,
        preserve_components,
        preserve_semantic_importance_threshold,
    )


def analyze_unimodal_model_for_surgery(model: nn.Module) -> Dict[str, Any]:
    """
    Convenience function to analyze a unimodal model for surgery.

    Args:
        model: The model to analyze

    Returns:
        Dictionary containing analysis results
    """
    surgery_system = get_unimodal_model_surgery_system()
    return surgery_system.analyze_unimodal_model_for_surgery(model)


__all__ = [
    "UnimodalModelSurgerySystem",
    "UnimodalComponentType",
    "UnimodalSurgicalComponent",
    "get_unimodal_model_surgery_system",
    "apply_unimodal_model_surgery",
    "analyze_unimodal_model_for_surgery",
]
