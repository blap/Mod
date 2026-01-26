"""
Multimodal Model Surgery System for Inference-PIO

This module extends the basic Model Surgery system to handle multimodal models,
specifically focusing on vision-language models like Qwen3-VL-2B. It provides
specialized techniques for pruning and modifying components that handle
cross-modal interactions while maintaining integrity across different modalities.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import copy

from .model_surgery import ModelSurgerySystem, ComponentType, SurgicalComponent
from .unimodal_model_surgery import (
    UnimodalModelSurgerySystem,
    UnimodalComponentType,
    UnimodalSurgicalComponent,
    get_unimodal_model_surgery_system,
    apply_unimodal_model_surgery,
    analyze_unimodal_model_for_surgery
)


logger = logging.getLogger(__name__)


class MultimodalComponentType(Enum):
    """Extended types of multimodal model components that can be surgically modified."""
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
    
    # Multimodal-specific components
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    MODALITY_ALIGNMENT = "modality_alignment"
    MULTIMODAL_FUSION = "multimodal_fusion"
    VISUAL_ENCODER = "visual_encoder"
    TEXT_ENCODER = "text_encoder"
    AUDIO_ENCODER = "audio_encoder"
    MODALITY_ADAPTER = "modality_adapter"
    
    # Vision-specific components
    VISION_TRANSFORMER_BLOCK = "vision_transformer_block"
    CONVOLUTIONAL_VISUAL_FEATURES = "convolutional_visual_features"
    VISION_PATCH_EMBEDDING = "vision_patch_embedding"
    
    # Language-specific components
    TEXT_EMBEDDING_LAYER = "text_embedding_layer"
    LANGUAGE_TRANSFORMER_BLOCK = "language_transformer_block"
    
    # Cross-modal components
    CROSS_MODAL_PROJECTION = "cross_modal_projection"
    MODALITY_GATE = "modality_gate"
    CROSS_MODAL_NORMALIZATION = "cross_modal_normalization"


@dataclass
class MultimodalSurgicalComponent(SurgicalComponent):
    """Extends SurgicalComponent with multimodal-specific properties."""
    modalities_involved: List[str] = None  # Which modalities this component handles
    cross_modal_interactions: int = 0     # Number of cross-modal interactions
    modality_balance: float = 1.0         # Balance between modalities (0.0-1.0)


class MultimodalModelSurgerySystem(ModelSurgerySystem):
    """
    Advanced Model Surgery System specifically designed for multimodal models.
    Focuses on maintaining integrity across different modalities while optimizing performance.
    """

    def __init__(self):
        super().__init__()
        self.multimodal_components: Dict[str, MultimodalSurgicalComponent] = {}
        self.modality_dependencies: Dict[str, Set[str]] = defaultdict(set)

    def identify_removable_components(self, model: nn.Module) -> List[MultimodalSurgicalComponent]:
        """
        Identify multimodal components that can be safely removed during inference.

        Args:
            model: The neural network model to analyze

        Returns:
            List of removable multimodal components with their details
        """
        removable_components = []

        # Walk through all modules in the model
        for name, module in model.named_modules():
            component_type = self._classify_multimodal_module(module)

            if component_type:
                # Determine if this component can be safely removed
                can_remove, reason, priority = self._can_safely_remove_multimodal(
                    module, component_type, name
                )

                if can_remove:
                    # Analyze multimodal properties
                    modalities_involved = self._analyze_modalities(module, name)
                    cross_modal_interactions = self._count_cross_modal_interactions(module)
                    modality_balance = self._calculate_modality_balance(module)

                    surgical_component = MultimodalSurgicalComponent(
                        name=name,
                        module=module,
                        type=component_type,
                        original_state=None,  # Will be set during removal
                        is_removed=False,
                        removal_reason=reason,
                        priority=priority,
                        modalities_involved=modalities_involved,
                        cross_modal_interactions=cross_modal_interactions,
                        modality_balance=modality_balance
                    )
                    removable_components.append(surgical_component)

        # Sort by priority (lower numbers = higher priority for removal)
        removable_components.sort(key=lambda x: x.priority)

        return removable_components

    def _classify_multimodal_module(self, module: nn.Module) -> Optional[MultimodalComponentType]:
        """
        Classify a module to determine its multimodal type for surgical purposes.

        Args:
            module: The module to classify

        Returns:
            MultimodalComponentType if the module is of interest, None otherwise
        """
        # First, try the base classification
        base_type = self._classify_module(module)
        if base_type:
            # Map base type to multimodal type
            return MultimodalComponentType(base_type.value)

        # Check for multimodal-specific components
        module_class_name = module.__class__.__name__.lower()
        
        # Vision-specific components
        if 'vision' in module_class_name and 'transformer' in module_class_name:
            return MultimodalComponentType.VISION_TRANSFORMER_BLOCK
        elif 'patch' in module_class_name and 'embed' in module_class_name:
            return MultimodalComponentType.VISION_PATCH_EMBEDDING
        elif isinstance(module, nn.Conv2d) and 'visual' in module_class_name:
            return MultimodalComponentType.CONVOLUTIONAL_VISUAL_FEATURES
            
        # Language-specific components
        elif 'embedding' in module_class_name and 'text' in module_class_name:
            return MultimodalComponentType.TEXT_EMBEDDING_LAYER
        elif 'transformer' in module_class_name and 'language' in module_class_name:
            return MultimodalComponentType.LANGUAGE_TRANSFORMER_BLOCK
            
        # Cross-modal components
        elif 'projection' in module_class_name and ('cross' in module_class_name or 'multi' in module_class_name):
            return MultimodalComponentType.CROSS_MODAL_PROJECTION
        elif 'gate' in module_class_name and ('modal' in module_class_name or 'cross' in module_class_name):
            return MultimodalComponentType.MODALITY_GATE
        elif 'norm' in module_class_name and ('cross' in module_class_name or 'multi' in module_class_name):
            return MultimodalComponentType.CROSS_MODAL_NORMALIZATION

        return None

    def _can_safely_remove_multimodal(self, module: nn.Module,
                                      component_type: MultimodalComponentType,
                                      name: str) -> Tuple[bool, str, int]:
        """
        Determine if a multimodal component can be safely removed during inference.

        Args:
            module: The module to evaluate
            component_type: The type of multimodal component
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
            # Component type doesn't exist in base enum, continue with multimodal-specific logic
            pass

        # Special handling for multimodal components
        if component_type == MultimodalComponentType.VISION_TRANSFORMER_BLOCK:
            # Vision transformer blocks can sometimes be pruned if not critical
            return True, "Vision transformer block can be simplified during inference", 8

        elif component_type == MultimodalComponentType.LANGUAGE_TRANSFORMER_BLOCK:
            # Language transformer blocks can sometimes be pruned if not critical
            return True, "Language transformer block can be simplified during inference", 8

        elif component_type == MultimodalComponentType.VISION_PATCH_EMBEDDING:
            # Patch embeddings are important but may have redundant features
            return True, "Vision patch embedding can be simplified during inference", 7

        elif component_type == MultimodalComponentType.TEXT_EMBEDDING_LAYER:
            # Text embeddings are important but may have redundant features
            return True, "Text embedding layer can be simplified during inference", 7

        elif component_type == MultimodalComponentType.CROSS_MODAL_PROJECTION:
            # Cross-modal projections can sometimes be simplified
            return True, "Cross-modal projection can be simplified during inference", 6

        elif component_type == MultimodalComponentType.MODALITY_GATE:
            # Modality gates can sometimes be simplified based on input modality
            return True, "Modality gate can be simplified during inference", 6

        elif component_type == MultimodalComponentType.CROSS_MODAL_NORMALIZATION:
            # Cross-modal normalization can sometimes be simplified
            return True, "Cross-modal normalization can be simplified during inference", 6

        # Default: don't remove
        return False, "Component is essential for multimodal inference", 100

    def _analyze_modalities(self, module: nn.Module, name: str) -> List[str]:
        """
        Analyze which modalities a module handles.

        Args:
            module: The module to analyze
            name: The name/path of the module

        Returns:
            List of modalities involved
        """
        modalities = []
        
        # Analyze based on module name
        name_lower = name.lower()
        if 'vision' in name_lower or 'visual' in name_lower or 'img' in name_lower:
            modalities.append('vision')
        if 'text' in name_lower or 'lang' in name_lower or 'sentence' in name_lower:
            modalities.append('text')
        if 'audio' in name_lower or 'speech' in name_lower:
            modalities.append('audio')
            
        # If no specific modality detected, assume it's a general component
        if not modalities:
            modalities.append('general')
            
        return modalities

    def _count_cross_modal_interactions(self, module: nn.Module) -> int:
        """
        Count the number of cross-modal interactions in a module.

        Args:
            module: The module to analyze

        Returns:
            Number of cross-modal interactions
        """
        # This is a simplified implementation
        # In a real implementation, this would analyze the module's architecture
        # to determine how many cross-modal connections exist
        return 0

    def _calculate_modality_balance(self, module: nn.Module) -> float:
        """
        Calculate the balance between different modalities in a module.

        Args:
            module: The module to analyze

        Returns:
            Balance score (0.0-1.0)
        """
        # This is a simplified implementation
        # In a real implementation, this would analyze the module's architecture
        # to determine how balanced the treatment of different modalities is
        return 1.0

    def perform_multimodal_surgery(self, model: nn.Module, 
                                   components_to_remove: Optional[List[str]] = None,
                                   preserve_components: Optional[List[str]] = None,
                                   preserve_modalities: Optional[List[str]] = None) -> nn.Module:
        """
        Perform multimodal model surgery by removing specified components.

        Args:
            model: The model to perform surgery on
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable
            preserve_modalities: Modalities to preserve (e.g., ['vision', 'text'])

        Returns:
            Modified model with components removed
        """
        # Identify components to remove if not specified
        if components_to_remove is None:
            all_removable = self.identify_removable_components(model)
            components_to_remove = [comp.name for comp in all_removable]

        # Apply preservation filters
        if preserve_components:
            components_to_remove = [name for name in components_to_remove
                                  if name not in preserve_components]

        # Filter based on modalities to preserve
        if preserve_modalities:
            filtered_components = []
            for name in components_to_remove:
                # Check if this component affects a preserved modality
                should_preserve = False
                for modality in preserve_modalities:
                    if modality.lower() in name.lower():
                        should_preserve = True
                        break
                
                if not should_preserve:
                    filtered_components.append(name)
            
            components_to_remove = filtered_components

        # Perform the actual surgery using the base method
        return self.perform_surgery(model, components_to_remove, preserve_components)

    def analyze_multimodal_model_for_surgery(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze a multimodal model to provide recommendations for surgery.

        Args:
            model: The model to analyze

        Returns:
            Dictionary containing multimodal analysis results
        """
        analysis = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'total_modules': 0,
            'removable_components': {},
            'potential_savings': 0,
            'recommendations': [],
            'modality_distribution': defaultdict(int),
            'cross_modal_analysis': {}
        }

        # Count total modules
        for name, module in model.named_modules():
            analysis['total_modules'] += 1
            # Count by modality
            name_lower = name.lower()
            if 'vision' in name_lower or 'visual' in name_lower or 'img' in name_lower:
                analysis['modality_distribution']['vision'] += 1
            elif 'text' in name_lower or 'lang' in name_lower or 'sentence' in name_lower:
                analysis['modality_distribution']['text'] += 1
            elif 'audio' in name_lower or 'speech' in name_lower:
                analysis['modality_distribution']['audio'] += 1
            else:
                analysis['modality_distribution']['general'] += 1

        # Identify removable components
        removable = self.identify_removable_components(model)

        for comp in removable:
            comp_type = comp.type.value
            if comp_type not in analysis['removable_components']:
                analysis['removable_components'][comp_type] = []

            analysis['removable_components'][comp_type].append({
                'name': comp.name,
                'reason': comp.removal_reason,
                'priority': comp.priority,
                'modalities_involved': comp.modalities_involved,
                'cross_modal_interactions': comp.cross_modal_interactions
            })

        # Generate recommendations
        if removable:
            analysis['recommendations'].append(
                f"Found {len(removable)} potentially removable multimodal components that could reduce model size"
            )

            # Count by type
            type_counts = defaultdict(int)
            for comp in removable:
                type_counts[comp.type.value] += 1

            for comp_type, count in type_counts.items():
                analysis['recommendations'].append(
                    f"- {count} {comp_type} components"
                )
                
            # Modality-specific recommendations
            for modality, count in analysis['modality_distribution'].items():
                if modality != 'general':
                    analysis['recommendations'].append(
                        f"- {count} {modality}-related components"
                    )
        else:
            analysis['recommendations'].append("No obvious candidates for surgical removal found")

        return analysis


# Global instance for convenience
_multimodal_model_surgery_system = MultimodalModelSurgerySystem()


def get_multimodal_model_surgery_system() -> MultimodalModelSurgerySystem:
    """
    Get the global multimodal model surgery system instance.

    Returns:
        MultimodalModelSurgerySystem instance
    """
    return _multimodal_model_surgery_system


def apply_multimodal_model_surgery(model: nn.Module, 
                                  components_to_remove: Optional[List[str]] = None,
                                  preserve_components: Optional[List[str]] = None,
                                  preserve_modalities: Optional[List[str]] = None) -> nn.Module:
    """
    Convenience function to apply multimodal model surgery to a model.

    Args:
        model: The model to perform surgery on
        components_to_remove: Specific components to remove (None means auto-detect)
        preserve_components: Components to preserve even if they're removable
        preserve_modalities: Modalities to preserve (e.g., ['vision', 'text'])

    Returns:
        Modified model with surgery applied
    """
    surgery_system = get_multimodal_model_surgery_system()
    return surgery_system.perform_multimodal_surgery(
        model, components_to_remove, preserve_components, preserve_modalities
    )


def analyze_multimodal_model_for_surgery(model: nn.Module) -> Dict[str, Any]:
    """
    Convenience function to analyze a multimodal model for surgery.

    Args:
        model: The model to analyze

    Returns:
        Dictionary containing analysis results
    """
    surgery_system = get_multimodal_model_surgery_system()
    return surgery_system.analyze_multimodal_model_for_surgery(model)


__all__ = [
    "MultimodalModelSurgerySystem",
    "MultimodalComponentType",
    "MultimodalSurgicalComponent",
    "get_multimodal_model_surgery_system",
    "apply_multimodal_model_surgery",
    "analyze_multimodal_model_for_surgery"
]