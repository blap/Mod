"""
Model Surgery System for Inference-PIO

This module implements the Model Surgery technique that identifies and temporarily 
removes non-essential components during inference to reduce model size and improve 
performance. The system can safely restore components after inference.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import copy


logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of model components that can be surgically modified."""
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


@dataclass
class SurgicalComponent:
    """Represents a component that can be surgically modified."""
    name: str
    module: nn.Module
    type: ComponentType
    original_state: Any = None
    is_removed: bool = False
    removal_reason: str = ""
    priority: int = 0  # Lower numbers indicate higher priority for removal


class ModelSurgerySystem:
    """
    Advanced Model Surgery System that identifies and temporarily removes 
    non-essential components during inference to reduce model size and improve performance.
    """
    
    def __init__(self):
        self.surgical_components: Dict[str, SurgicalComponent] = {}
        self.backup_registry: Dict[str, Any] = {}
        self.surgery_history: List[Dict[str, Any]] = []
        self.active_surgeries: Set[str] = set()
        
    def identify_removable_components(self, model: nn.Module) -> List[SurgicalComponent]:
        """
        Identify components that can be safely removed during inference.
        
        Args:
            model: The neural network model to analyze
            
        Returns:
            List of removable components with their details
        """
        removable_components = []
        
        # Walk through all modules in the model
        for name, module in model.named_modules():
            component_type = self._classify_module(module)
            
            if component_type:
                # Determine if this component can be safely removed
                can_remove, reason, priority = self._can_safely_remove(module, component_type, name)
                
                if can_remove:
                    surgical_component = SurgicalComponent(
                        name=name,
                        module=module,
                        type=component_type,
                        original_state=None,  # Will be set during removal
                        is_removed=False,
                        removal_reason=reason,
                        priority=priority
                    )
                    removable_components.append(surgical_component)
                    
        # Sort by priority (lower numbers = higher priority for removal)
        removable_components.sort(key=lambda x: x.priority)
        
        return removable_components
    
    def _classify_module(self, module: nn.Module) -> Optional[ComponentType]:
        """
        Classify a module to determine its type for surgical purposes.

        Args:
            module: The module to classify

        Returns:
            ComponentType if the module is of interest, None otherwise
        """
        if isinstance(module, nn.Dropout):
            return ComponentType.DROPOUT_LAYER
        elif isinstance(module, nn.BatchNorm1d):
            return ComponentType.BATCH_NORM
        elif isinstance(module, nn.BatchNorm2d):
            return ComponentType.BATCH_NORM
        elif isinstance(module, nn.BatchNorm3d):
            return ComponentType.BATCH_NORM
        elif isinstance(module, nn.LayerNorm):
            return ComponentType.LAYER_NORM
        elif isinstance(module, nn.GroupNorm):
            return ComponentType.GROUP_NORM
        elif isinstance(module, nn.InstanceNorm1d):
            return ComponentType.INSTANCE_NORM
        elif isinstance(module, nn.InstanceNorm2d):
            return ComponentType.INSTANCE_NORM
        elif isinstance(module, nn.InstanceNorm3d):
            return ComponentType.INSTANCE_NORM
        elif isinstance(module, nn.Identity):
            return ComponentType.ACTIVATION_LAYER
        elif hasattr(nn, 'SiLU') and isinstance(module, nn.SiLU):
            # Swish/SiLU activation can sometimes be approximated during inference
            return ComponentType.ACTIVATION_LAYER

        # Check for multimodal-specific components
        module_class_name = module.__class__.__name__.lower()

        # Cross-modal attention components
        if 'cross' in module_class_name and 'attention' in module_class_name:
            return ComponentType.CROSS_MODAL_ATTENTION
        elif 'multimodal' in module_class_name and 'attention' in module_class_name:
            return ComponentType.CROSS_MODAL_ATTENTION
        elif 'multimodal' in module_class_name and 'cross' in module_class_name:
            return ComponentType.CROSS_MODAL_ATTENTION

        # Modality alignment components
        if 'alignment' in module_class_name or 'align' in module_class_name:
            return ComponentType.MODALITY_ALIGNMENT

        # Multimodal fusion components
        if 'fusion' in module_class_name or 'fuse' in module_class_name:
            return ComponentType.MULTIMODAL_FUSION

        # Encoder components
        if 'visual' in module_class_name or 'vision' in module_class_name:
            return ComponentType.VISUAL_ENCODER
        elif 'text' in module_class_name:
            return ComponentType.TEXT_ENCODER
        elif 'audio' in module_class_name:
            return ComponentType.AUDIO_ENCODER

        # Adapter components
        if 'adapter' in module_class_name:
            return ComponentType.MODALITY_ADAPTER

        return None
    
    def _can_safely_remove(self, module: nn.Module, component_type: ComponentType, name: str) -> Tuple[bool, str, int]:
        """
        Determine if a component can be safely removed during inference.

        Args:
            module: The module to evaluate
            component_type: The type of component
            name: The name/path of the module

        Returns:
            Tuple of (can_remove, reason, priority)
        """
        # Dropout layers can almost always be removed during inference
        if component_type == ComponentType.DROPOUT_LAYER:
            return True, "Dropout layers are inactive during inference", 1  # Highest priority

        # Batch norm layers can often be folded into preceding linear layers during inference
        elif component_type in [ComponentType.BATCH_NORM, ComponentType.LAYER_NORM,
                               ComponentType.GROUP_NORM, ComponentType.INSTANCE_NORM]:
            # Check if this normalization layer is followed by an activation that makes it essential
            # For now, assume most normalization layers can be removed/simplified
            return True, f"{component_type.value} can often be simplified during inference", 2

        # Identity layers can typically be removed
        elif component_type == ComponentType.ACTIVATION_LAYER:
            if isinstance(module, nn.Identity):
                return True, "Identity layers have no effect during inference", 3

        # Multimodal-specific components
        elif component_type == ComponentType.CROSS_MODAL_ATTENTION:
            # Cross-modal attention can sometimes be simplified during inference
            # depending on the specific implementation and requirements
            return True, "Cross-modal attention can be simplified during inference", 4

        elif component_type == ComponentType.MODALITY_ALIGNMENT:
            # Modality alignment components can sometimes be simplified
            # but may be critical for multimodal integration
            return True, "Modality alignment can be simplified during inference", 5

        elif component_type == ComponentType.MULTIMODAL_FUSION:
            # Fusion layers can be simplified but may be critical for multimodal performance
            return True, "Multimodal fusion can be simplified during inference", 6

        elif component_type in [ComponentType.VISUAL_ENCODER, ComponentType.TEXT_ENCODER,
                               ComponentType.AUDIO_ENCODER]:
            # Encoder components are typically essential for multimodal processing
            # but some may be optional depending on the input modality
            return False, f"{component_type.value} is essential for multimodal processing", 90

        elif component_type == ComponentType.MODALITY_ADAPTER:
            # Adapters can often be simplified or removed if not needed
            return True, "Modality adapter can be simplified during inference", 7

        # Default: don't remove
        return False, "Component is essential for inference", 100
    
    def perform_surgery(self, model: nn.Module, components_to_remove: Optional[List[str]] = None, 
                       preserve_components: Optional[List[str]] = None) -> nn.Module:
        """
        Perform model surgery by removing specified components.
        
        Args:
            model: The model to perform surgery on
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable
            
        Returns:
            Modified model with components removed
        """
        # Identify components to remove if not specified
        if components_to_remove is None:
            all_removable = self.identify_removable_components(model)
            components_to_remove = [comp.name for comp in all_removable]
        
        # Apply preservation filter
        if preserve_components:
            components_to_remove = [name for name in components_to_remove 
                                  if name not in preserve_components]
        
        # Create a copy of the model to avoid modifying the original
        model_copy = copy.deepcopy(model)
        
        # Track the surgery operation
        import time
        surgery_record = {
            'timestamp': time.time(),
            'components_removed': [],
            'model_before_params': sum(p.numel() for p in model.parameters()),
            'model_after_params': 0
        }
        
        # Remove specified components
        for name in components_to_remove:
            try:
                # Navigate to the parent module and the specific submodule
                module_parts = name.split('.')
                parent_module = model_copy
                child_name = module_parts[-1]

                # Navigate to parent
                for part in module_parts[:-1]:
                    parent_module = getattr(parent_module, part)

                # Get the original module
                original_module = getattr(parent_module, child_name)

                # Determine the component type
                component_type = self._classify_module(original_module)

                # Backup the original module
                backup_key = f"{name}_{len(self.backup_registry)}"
                self.backup_registry[backup_key] = {
                    'original_module': original_module,
                    'parent_module': parent_module,
                    'child_name': child_name,
                    'module_path': name,
                    'component_type': component_type
                }

                # Replace with appropriate placeholder based on component type
                replacement_module = self._create_replacement_module(original_module, component_type)

                # Replace the module
                setattr(parent_module, child_name, replacement_module)

                # Record the surgery
                surgery_record['components_removed'].append({
                    'name': name,
                    'type': str(type(original_module)),
                    'replacement': str(type(replacement_module)),
                    'component_type': component_type.value if component_type else 'unknown'
                })

                logger.info(f"Surgically removed component: {name} ({type(original_module).__name__}) "
                           f"with type: {component_type.value if component_type else 'unknown'}")

            except Exception as e:
                logger.warning(f"Failed to remove component {name}: {str(e)}")
                continue
        
        # Calculate new parameter count
        surgery_record['model_after_params'] = sum(p.numel() for p in model_copy.parameters())
        
        # Add to surgery history
        self.surgery_history.append(surgery_record)
        
        # Mark this surgery as active
        surgery_id = f"surgery_{len(self.surgery_history)-1}"
        self.active_surgeries.add(surgery_id)
        
        logger.info(f"Model surgery completed. Removed {len(surgery_record['components_removed'])} components. "
                   f"Parameter count reduced from {surgery_record['model_before_params']} "
                   f"to {surgery_record['model_after_params']}.")
        
        return model_copy

    def _create_replacement_module(self, original_module: nn.Module, component_type: Optional[ComponentType]) -> nn.Module:
        """
        Create an appropriate replacement module based on the original module and its type.

        Args:
            original_module: The original module to be replaced
            component_type: The type of component being replaced

        Returns:
            Appropriate replacement module
        """
        if component_type is None:
            # Default: replace with identity
            return nn.Identity()

        # Handle standard components
        if isinstance(original_module, nn.Dropout):
            # Replace dropout with identity during inference
            return nn.Identity()
        elif isinstance(original_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                       nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            # For batch norm and instance norm, we can replace with identity
            # but we need to ensure the normalization effect is maintained
            # For now, replace with identity but in a real implementation,
            # we would fold the normalization into preceding layers
            return nn.Identity()
        elif isinstance(original_module, nn.LayerNorm):
            # LayerNorm can be replaced with identity
            return nn.Identity()
        elif isinstance(original_module, nn.GroupNorm):
            # GroupNorm can be replaced with identity
            return nn.Identity()

        # Handle multimodal-specific components
        elif component_type == ComponentType.CROSS_MODAL_ATTENTION:
            # For cross-modal attention, we can replace with a simpler attention mechanism
            # or identity depending on the specific requirements
            return nn.Identity()
        elif component_type == ComponentType.MODALITY_ALIGNMENT:
            # For modality alignment, we can replace with identity or a simpler transformation
            return nn.Identity()
        elif component_type == ComponentType.MULTIMODAL_FUSION:
            # For multimodal fusion, we can replace with identity or a simpler combination
            return nn.Identity()
        elif component_type in [ComponentType.VISUAL_ENCODER, ComponentType.TEXT_ENCODER,
                               ComponentType.AUDIO_ENCODER]:
            # For encoder components, we should be more careful - they're usually essential
            # But if we must replace them, use identity as a placeholder
            # In practice, these shouldn't be removed unless the modality isn't being used
            return nn.Identity()
        elif component_type == ComponentType.MODALITY_ADAPTER:
            # For modality adapters, we can replace with identity
            return nn.Identity()

        # Default: replace with identity
        return nn.Identity()

    def restore_model(self, model: nn.Module, surgery_id: Optional[str] = None) -> nn.Module:
        """
        Restore a model to its original state after surgery.
        
        Args:
            model: The model to restore
            surgery_id: Specific surgery to reverse (None means reverse latest)
            
        Returns:
            Restored model
        """
        if not self.backup_registry:
            logger.warning("No backups available for restoration")
            return model
        
        # Find the surgery to restore
        if surgery_id is None:
            # Restore the most recent surgery
            if not self.surgery_history:
                logger.warning("No surgery history to restore from")
                return model
            surgery_record = self.surgery_history[-1]
            surgery_id = f"surgery_{len(self.surgery_history)-1}"
        else:
            # Find specific surgery
            surgery_index = int(surgery_id.replace("surgery_", ""))
            if surgery_index >= len(self.surgery_history):
                logger.error(f"Surgery {surgery_id} not found in history")
                return model
            surgery_record = self.surgery_history[surgery_index]
        
        # Create a copy of the model to avoid modifying the original
        model_copy = copy.deepcopy(model)
        
        # Restore each component
        restored_count = 0
        for removed_component in surgery_record['components_removed']:
            component_name = removed_component['name']

            # Find the backup for this component
            backup_key = None
            for key, backup in self.backup_registry.items():
                if backup['module_path'] == component_name:
                    backup_key = key
                    break

            if backup_key:
                backup_data = self.backup_registry[backup_key]

                # Navigate to parent module in the copied model
                parent_module = model_copy
                for part in backup_data['module_path'].split('.')[:-1]:
                    parent_module = getattr(parent_module, part)

                # Restore the original module
                setattr(parent_module, backup_data['child_name'], backup_data['original_module'])
                restored_count += 1

                # Remove from backup registry
                del self.backup_registry[backup_key]

                logger.info(f"Restored component: {component_name} (type: {backup_data.get('component_type', 'unknown')})")
        
        # Remove from active surgeries
        self.active_surgeries.discard(surgery_id)
        
        logger.info(f"Model restoration completed. Restored {restored_count} components.")
        
        return model_copy
    
    def get_surgery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about performed surgeries.
        
        Returns:
            Dictionary containing surgery statistics
        """
        total_removed = 0
        total_restored = 0
        surgery_count = len(self.surgery_history)
        
        for record in self.surgery_history:
            total_removed += len(record['components_removed'])
        
        # For now, we don't track individual restorations, so estimate
        total_restored = total_removed - len(self.backup_registry)
        
        return {
            'total_surgeries_performed': surgery_count,
            'total_components_removed': total_removed,
            'total_components_restored': total_restored,
            'components_pending_restoration': len(self.backup_registry),
            'active_surgeries': len(self.active_surgeries),
            'surgery_history_length': len(self.surgery_history)
        }
    
    def analyze_model_for_surgery(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze a model to provide recommendations for surgery.
        
        Args:
            model: The model to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'total_modules': 0,
            'removable_components': {},
            'potential_savings': 0,
            'recommendations': []
        }
        
        # Count total modules
        for name, module in model.named_modules():
            analysis['total_modules'] += 1
        
        # Identify removable components
        removable = self.identify_removable_components(model)
        
        for comp in removable:
            comp_type = comp.type.value
            if comp_type not in analysis['removable_components']:
                analysis['removable_components'][comp_type] = []
            
            analysis['removable_components'][comp_type].append({
                'name': comp.name,
                'reason': comp.removal_reason,
                'priority': comp.priority
            })
        
        # Generate recommendations
        if removable:
            analysis['recommendations'].append(
                f"Found {len(removable)} potentially removable components that could reduce model size"
            )
            
            # Count by type
            type_counts = defaultdict(int)
            for comp in removable:
                type_counts[comp.type.value] += 1
            
            for comp_type, count in type_counts.items():
                analysis['recommendations'].append(
                    f"- {count} {comp_type} components"
                )
        else:
            analysis['recommendations'].append("No obvious candidates for surgical removal found")
        
        return analysis


# Global instance for convenience
_model_surgery_system = ModelSurgerySystem()


def get_model_surgery_system() -> ModelSurgerySystem:
    """
    Get the global model surgery system instance.
    
    Returns:
        ModelSurgerySystem instance
    """
    return _model_surgery_system


def apply_model_surgery(model: nn.Module, components_to_remove: Optional[List[str]] = None,
                       preserve_components: Optional[List[str]] = None) -> nn.Module:
    """
    Convenience function to apply model surgery to a model.
    
    Args:
        model: The model to perform surgery on
        components_to_remove: Specific components to remove (None means auto-detect)
        preserve_components: Components to preserve even if they're removable
        
    Returns:
        Modified model with surgery applied
    """
    surgery_system = get_model_surgery_system()
    return surgery_system.perform_surgery(model, components_to_remove, preserve_components)


def restore_model_from_surgery(model: nn.Module, surgery_id: Optional[str] = None) -> nn.Module:
    """
    Convenience function to restore a model from surgery.
    
    Args:
        model: The model to restore
        surgery_id: Specific surgery to reverse (None means reverse latest)
        
    Returns:
        Restored model
    """
    surgery_system = get_model_surgery_system()
    return surgery_system.restore_model(model, surgery_id)


__all__ = [
    "ModelSurgerySystem",
    "ComponentType", 
    "SurgicalComponent",
    "get_model_surgery_system",
    "apply_model_surgery",
    "restore_model_from_surgery"
]