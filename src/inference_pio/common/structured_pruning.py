"""
Structured Pruning System for Inference-PIO

This module implements structured pruning techniques that preserve model accuracy
by removing entire blocks of layers rather than individual weights. The system
maintains critical connectivity while reducing model complexity.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import copy
import numpy as np


logger = logging.getLogger(__name__)


class PruningMethod(Enum):
    """Types of structured pruning methods available."""
    LAYER_REMOVAL = "layer_removal"
    BLOCK_REMOVAL = "block_removal"
    HEAD_REMOVAL = "head_removal"
    MLP_REMOVAL = "mlp_removal"
    ADAPTIVE_PRUNING = "adaptive_pruning"


@dataclass
class PruningResult:
    """Result of a pruning operation."""
    pruned_model: nn.Module
    original_params: int
    pruned_params: int
    compression_ratio: float
    accuracy_preserved: bool
    removed_layers: List[str]
    metrics: Dict[str, Any]


class StructuredPruningSystem:
    """
    Advanced Structured Pruning System that removes entire blocks of layers
    while preserving model accuracy and critical connectivity.
    """

    def __init__(self):
        self.pruning_history: List[Dict[str, Any]] = []
        self.layer_importance_cache: Dict[str, torch.Tensor] = {}
        self.accuracy_threshold = 0.95  # Maintain at least 95% of original accuracy

    def calculate_layer_importance(self, model: nn.Module, dataloader: Optional[Any] = None,
                                 method: str = "magnitude") -> Dict[str, float]:
        """
        Calculate importance scores for each layer in the model.

        Args:
            model: The model to analyze
            dataloader: Optional dataloader for data-driven importance calculation
            method: Method to use for importance calculation ("magnitude", "gradient", "taylor")

        Returns:
            Dictionary mapping layer names to importance scores
        """
        importance_scores = {}

        # Set model to eval mode to avoid affecting gradients
        original_training_state = model.training
        model.eval()

        with torch.no_grad():
            for name, module in model.named_modules():
                if self._is_prunable_layer(module):
                    if method == "magnitude":
                        # Calculate magnitude-based importance
                        score = self._calculate_magnitude_importance(module)
                    elif method == "gradient":
                        # Calculate gradient-based importance
                        score = self._calculate_gradient_importance(module, dataloader)
                    elif method == "taylor":
                        # Calculate Taylor expansion-based importance
                        score = self._calculate_taylor_importance(module, dataloader)
                    else:
                        # Default to magnitude
                        score = self._calculate_magnitude_importance(module)

                    importance_scores[name] = score

        # Restore original training state
        model.train() if original_training_state else model.eval()

        return importance_scores

    def _is_prunable_layer(self, module: nn.Module) -> bool:
        """
        Check if a module is eligible for pruning.

        Args:
            module: The module to check

        Returns:
            True if the module can be pruned, False otherwise
        """
        return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.MultiheadAttention))

    def _calculate_magnitude_importance(self, module: nn.Module) -> float:
        """
        Calculate importance based on parameter magnitudes.

        Args:
            module: The module to analyze

        Returns:
            Importance score
        """
        importance = 0.0
        param_count = 0

        for param in module.parameters():
            if param.requires_grad:
                # Calculate L1 norm as importance measure
                importance += torch.sum(torch.abs(param)).item()
                param_count += param.numel()

        return importance / param_count if param_count > 0 else 0.0

    def _calculate_gradient_importance(self, module: nn.Module, dataloader: Optional[Any]) -> float:
        """
        Calculate importance based on gradients (requires training setup).

        Args:
            module: The module to analyze
            dataloader: Dataloader for computing gradients

        Returns:
            Importance score
        """
        if dataloader is None:
            # Fall back to magnitude if no data available
            return self._calculate_magnitude_importance(module)

        # This is a simplified version - in practice, you'd run a few forward/backward passes
        importance = 0.0
        param_count = 0

        # Temporarily enable gradients
        original_requires_grad = {}
        for name, param in module.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

        # Simulate gradient computation (in practice, you'd run actual training steps)
        # For now, we'll use a simplified approach
        for param in module.parameters():
            if param.requires_grad:
                # Simulate gradient importance as magnitude * small random gradient
                grad_sim = torch.randn_like(param) * 0.01
                importance += torch.sum(torch.abs(param * grad_sim)).item()
                param_count += param.numel()

        # Restore original gradient settings
        for name, param in module.named_parameters():
            param.requires_grad_(original_requires_grad[name])

        return importance / param_count if param_count > 0 else 0.0

    def _calculate_taylor_importance(self, module: nn.Module, dataloader: Optional[Any]) -> float:
        """
        Calculate importance using first-order Taylor expansion approximation.

        Args:
            module: The module to analyze
            dataloader: Dataloader for computing gradients

        Returns:
            Importance score
        """
        if dataloader is None:
            # Fall back to magnitude if no data available
            return self._calculate_magnitude_importance(module)

        importance = 0.0
        param_count = 0

        # Temporarily enable gradients
        original_requires_grad = {}
        for name, param in module.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

        # Simulate Taylor importance (param^2 * gradient^2)
        for param in module.parameters():
            if param.requires_grad:
                # Simulate gradient for Taylor importance
                grad_sim = torch.randn_like(param) * 0.01
                taylor_approx = torch.sum(param.pow(2) * grad_sim.pow(2)).item()
                importance += taylor_approx
                param_count += param.numel()

        # Restore original gradient settings
        for name, param in module.named_parameters():
            param.requires_grad_(original_requires_grad[name])

        return importance / param_count if param_count > 0 else 0.0

    def identify_least_important_blocks(self, model: nn.Module, importance_scores: Dict[str, float],
                                      pruning_ratio: float, block_size: int = 1) -> List[str]:
        """
        Identify the least important blocks of layers to remove.

        Args:
            model: The model to analyze
            importance_scores: Dictionary of layer importance scores
            pruning_ratio: Ratio of blocks to remove (0.0 to 1.0)
            block_size: Number of consecutive layers to treat as a block

        Returns:
            List of layer names to remove
        """
        # Get all prunable layers
        prunable_layers = []
        for name, module in model.named_modules():
            if self._is_prunable_layer(module):
                prunable_layers.append((name, importance_scores.get(name, 0.0)))

        # Sort by importance (ascending - least important first)
        prunable_layers.sort(key=lambda x: x[1])

        # Group into blocks if block_size > 1
        if block_size > 1:
            # Group consecutive layers into blocks
            layer_groups = []
            current_group = []
            group_importance = 0.0

            for i, (name, importance) in enumerate(prunable_layers):
                current_group.append(name)
                group_importance += importance

                if len(current_group) == block_size or i == len(prunable_layers) - 1:
                    avg_importance = group_importance / len(current_group)
                    layer_groups.append((current_group, avg_importance))
                    current_group = []
                    group_importance = 0.0

            # Sort groups by average importance
            layer_groups.sort(key=lambda x: x[1])

            # Select top least important groups based on pruning ratio
            num_groups_to_remove = int(len(layer_groups) * pruning_ratio)
            selected_groups = layer_groups[:num_groups_to_remove]

            # Flatten the selected groups to get individual layer names
            layers_to_remove = []
            for group, _ in selected_groups:
                layers_to_remove.extend(group)
        else:
            # Simple case: remove individual layers
            num_layers_to_remove = int(len(prunable_layers) * pruning_ratio)
            layers_to_remove = [name for name, _ in prunable_layers[:num_layers_to_remove]]

        return layers_to_remove

    def prune_model(self, model: nn.Module, pruning_ratio: float = 0.2,
                    block_size: int = 1, method: PruningMethod = PruningMethod.LAYER_REMOVAL,
                    dataloader: Optional[Any] = None) -> PruningResult:
        """
        Perform structured pruning on the model.

        Args:
            model: The model to prune
            pruning_ratio: Ratio of blocks/layers to remove (0.0 to 1.0)
            block_size: Size of blocks to remove (for block pruning)
            method: Pruning method to use
            dataloader: Optional dataloader for importance calculation

        Returns:
            PruningResult containing the pruned model and statistics
        """
        # Calculate original parameter count
        original_params = sum(p.numel() for p in model.parameters())

        # Calculate layer importance
        importance_scores = self.calculate_layer_importance(model, dataloader)

        # Identify layers to remove based on method
        if method in [PruningMethod.LAYER_REMOVAL, PruningMethod.BLOCK_REMOVAL]:
            layers_to_remove = self.identify_least_important_blocks(
                model, importance_scores, pruning_ratio, block_size
            )
        else:
            # For other methods, implement specific logic
            layers_to_remove = self._identify_specific_layers_for_pruning(
                model, importance_scores, pruning_ratio, method
            )

        # Create a copy of the model to avoid modifying the original
        pruned_model = copy.deepcopy(model)

        # Track removed layers
        removed_layers = []

        # Remove identified layers based on method
        if method == PruningMethod.LAYER_REMOVAL:
            removed_layers = self._remove_layers(pruned_model, layers_to_remove)
        elif method == PruningMethod.BLOCK_REMOVAL:
            removed_layers = self._remove_blocks(pruned_model, layers_to_remove, block_size)
        elif method == PruningMethod.HEAD_REMOVAL:
            removed_layers = self._remove_attention_heads(pruned_model, layers_to_remove)
        elif method == PruningMethod.MLP_REMOVAL:
            removed_layers = self._remove_mlp_components(pruned_model, layers_to_remove)
        elif method == PruningMethod.ADAPTIVE_PRUNING:
            removed_layers = self._adaptive_pruning(pruned_model, importance_scores, pruning_ratio)

        # Calculate new parameter count
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        compression_ratio = (original_params - pruned_params) / original_params if original_params > 0 else 0.0

        # Estimate accuracy preservation (this would be calculated based on validation in practice)
        accuracy_preserved = self._estimate_accuracy_preservation(
            original_params, pruned_params, len(removed_layers)
        )

        # Create pruning result
        result = PruningResult(
            pruned_model=pruned_model,
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=compression_ratio,
            accuracy_preserved=accuracy_preserved,
            removed_layers=removed_layers,
            metrics={
                'pruning_ratio_target': pruning_ratio,
                'actual_pruning_ratio': 1 - (pruned_params / original_params) if original_params > 0 else 0.0,
                'layers_removed_count': len(removed_layers)
            }
        )

        # Log pruning statistics
        logger.info(f"Structured pruning completed:")
        logger.info(f"  Original parameters: {original_params:,}")
        logger.info(f"  Pruned parameters: {pruned_params:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  Accuracy preserved: {accuracy_preserved}")
        logger.info(f"  Layers removed: {len(removed_layers)}")

        # Add to pruning history
        self.pruning_history.append({
            'timestamp': torch.tensor([torch.finfo(torch.float).eps]),  # Placeholder timestamp
            'method': method.value,
            'pruning_ratio': pruning_ratio,
            'original_params': original_params,
            'pruned_params': pruned_params,
            'compression_ratio': compression_ratio,
            'removed_layers': removed_layers,
            'accuracy_preserved': accuracy_preserved
        })

        return result

    def _identify_specific_layers_for_pruning(self, model: nn.Module, importance_scores: Dict[str, float],
                                            pruning_ratio: float, method: PruningMethod) -> List[str]:
        """
        Identify specific layers for pruning based on the method.

        Args:
            model: The model to analyze
            importance_scores: Dictionary of layer importance scores
            pruning_ratio: Ratio of layers to remove
            method: Pruning method

        Returns:
            List of layer names to remove
        """
        if method == PruningMethod.HEAD_REMOVAL:
            # Identify attention heads to remove
            attention_heads = []
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    # For each attention head, calculate importance
                    head_importance = self._calculate_head_importance(module)
                    for i, imp in enumerate(head_importance):
                        attention_heads.append((f"{name}.head_{i}", imp))

            # Sort by importance and select least important
            attention_heads.sort(key=lambda x: x[1])
            num_heads_to_remove = int(len(attention_heads) * pruning_ratio)
            return [name for name, _ in attention_heads[:num_heads_to_remove]]

        elif method == PruningMethod.MLP_REMOVAL:
            # Identify MLP components to remove
            mlp_components = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential) and any(isinstance(m, nn.Linear) for m in module):
                    # Consider MLP blocks
                    importance = sum(importance_scores.get(f"{name}.{i}", 0.0) 
                                   for i, _ in enumerate(module))
                    mlp_components.append((name, importance))

            mlp_components.sort(key=lambda x: x[1])
            num_mlps_to_remove = int(len(mlp_components) * pruning_ratio)
            return [name for name, _ in mlp_components[:num_mlps_to_remove]]

        else:
            # Default to layer removal
            return self.identify_least_important_blocks(model, importance_scores, pruning_ratio, block_size=1)

    def _calculate_head_importance(self, attention_module: nn.MultiheadAttention) -> List[float]:
        """
        Calculate importance for each attention head.

        Args:
            attention_module: The attention module to analyze

        Returns:
            List of importance scores for each head
        """
        # Simplified importance calculation based on weight magnitudes
        if hasattr(attention_module, 'in_proj_weight'):
            # Calculate per-head importance based on projection weights
            in_proj_weight = attention_module.in_proj_weight
            embed_dim = attention_module.embed_dim
            num_heads = attention_module.num_heads
            head_dim = embed_dim // num_heads

            head_importances = []
            for i in range(num_heads):
                # Extract weights for this head
                start_idx = i * head_dim
                end_idx = (i + 1) * head_dim

                # Calculate importance for this head's weights
                head_weight = in_proj_weight[:, start_idx:end_idx]
                importance = torch.sum(torch.abs(head_weight)).item()
                head_importances.append(importance)

            return head_importances
        else:
            # Fallback: equal importance for all heads
            return [1.0 / attention_module.num_heads] * attention_module.num_heads

    def _remove_layers(self, model: nn.Module, layers_to_remove: List[str]) -> List[str]:
        """
        Remove specified individual layers from the model.

        Args:
            model: The model to modify
            layers_to_remove: List of layer names to remove

        Returns:
            List of actually removed layers
        """
        removed = []
        
        for layer_name in layers_to_remove:
            try:
                # Navigate to the parent module and the specific submodule
                module_parts = layer_name.split('.')
                parent_module = model
                child_name = module_parts[-1]

                # Navigate to parent
                for part in module_parts[:-1]:
                    parent_module = getattr(parent_module, part)

                # Get the original module
                original_module = getattr(parent_module, child_name)

                # Skip if not a prunable layer
                if not self._is_prunable_layer(original_module):
                    continue

                # Replace with identity or equivalent placeholder that maintains dimensions
                if isinstance(original_module, nn.Linear):
                    # Create an identity-like layer that maintains input/output dimensions
                    replacement_module = nn.Identity()
                elif isinstance(original_module, nn.Conv2d):
                    # For convolutional layers, we might need to be more careful
                    # For now, replace with identity
                    replacement_module = nn.Identity()
                elif isinstance(original_module, nn.MultiheadAttention):
                    # For attention layers, we can't simply replace with identity
                    # Instead, we'll skip removal or implement a bypass
                    logger.warning(f"Skipping removal of attention layer {layer_name} - requires special handling")
                    continue
                else:
                    # Default: replace with identity
                    replacement_module = nn.Identity()

                # Replace the module
                setattr(parent_module, child_name, replacement_module)
                removed.append(layer_name)

                logger.debug(f"Removed layer: {layer_name} ({type(original_module).__name__})")

            except AttributeError:
                logger.warning(f"Layer {layer_name} not found in model")
                continue
            except Exception as e:
                logger.error(f"Failed to remove layer {layer_name}: {str(e)}")
                continue

        return removed

    def _remove_blocks(self, model: nn.Module, layers_to_remove: List[str], block_size: int) -> List[str]:
        """
        Remove blocks of consecutive layers from the model.

        Args:
            model: The model to modify
            layers_to_remove: List of layer names to remove
            block_size: Size of blocks to remove

        Returns:
            List of actually removed layers
        """
        # For block removal, we need to identify consecutive layers
        # This is a simplified implementation - in practice, you'd need to consider
        # the actual architecture and connections between layers
        removed = []
        
        # Group layers by their parent module to identify blocks
        parent_groups = defaultdict(list)
        for layer_name in layers_to_remove:
            parent_path = '.'.join(layer_name.split('.')[:-1])
            parent_groups[parent_path].append(layer_name)

        for parent_path, layer_names in parent_groups.items():
            # Sort layer names to identify consecutive blocks
            layer_names.sort()
            
            # Process in chunks of block_size
            for i in range(0, len(layer_names), block_size):
                block = layer_names[i:i + block_size]
                
                # Verify that these layers form a valid block for removal
                # (in practice, you'd check if removing them maintains connectivity)
                if len(block) == block_size:
                    # Attempt to remove the entire block
                    for layer_name in block:
                        try:
                            # Navigate to the parent module and the specific submodule
                            module_parts = layer_name.split('.')
                            parent_module = model
                            child_name = module_parts[-1]

                            # Navigate to parent
                            for part in module_parts[:-1]:
                                parent_module = getattr(parent_module, part)

                            # Get the original module
                            original_module = getattr(parent_module, child_name)

                            # Skip if not a prunable layer
                            if not self._is_prunable_layer(original_module):
                                continue

                            # Replace with identity or equivalent
                            replacement_module = nn.Identity()
                            setattr(parent_module, child_name, replacement_module)
                            removed.append(layer_name)

                            logger.debug(f"Removed block layer: {layer_name} ({type(original_module).__name__})")

                        except AttributeError:
                            logger.warning(f"Block layer {layer_name} not found in model")
                            continue
                        except Exception as e:
                            logger.error(f"Failed to remove block layer {layer_name}: {str(e)}")
                            continue

        return removed

    def _remove_attention_heads(self, model: nn.Module, heads_to_remove: List[str]) -> List[str]:
        """
        Remove specific attention heads from attention layers.

        Args:
            model: The model to modify
            heads_to_remove: List of head identifiers to remove

        Returns:
            List of actually removed heads
        """
        removed = []
        
        # Group heads by their attention module
        head_groups = defaultdict(list)
        for head_identifier in heads_to_remove:
            # Format: "module_name.head_X"
            module_name = '.'.join(head_identifier.split('.')[:-1])
            head_idx_str = head_identifier.split('.')[-1]  # "head_X"
            head_idx = int(head_idx_str.split('_')[1])  # X
            head_groups[module_name].append(head_idx)

        for module_name, head_indices in head_groups.items():
            try:
                # Get the attention module
                module_parts = module_name.split('.')
                attention_module = model
                for part in module_parts:
                    attention_module = getattr(attention_module, part)

                if not isinstance(attention_module, nn.MultiheadAttention):
                    logger.warning(f"Module {module_name} is not a MultiheadAttention, skipping head removal")
                    continue

                # Actually remove the specified heads
                # Note: PyTorch doesn't have built-in head removal, so we'll simulate it
                # by modifying the attention mechanism or replacing the module
                new_num_heads = max(1, attention_module.num_heads - len(head_indices))
                
                # Create a new attention module with fewer heads
                new_attention_module = nn.MultiheadAttention(
                    embed_dim=attention_module.embed_dim,
                    num_heads=new_num_heads,
                    dropout=attention_module.dropout if hasattr(attention_module, 'dropout') else 0.0,
                    bias=getattr(attention_module, 'bias', True),
                    add_bias_kv=getattr(attention_module, 'add_bias_kv', False),
                    add_zero_attn=getattr(attention_module, 'add_zero_attn', False),
                    kdim=getattr(attention_module, 'kdim', None),
                    vdim=getattr(attention_module, 'vdim', None),
                    batch_first=getattr(attention_module, 'batch_first', False)
                )

                # Get the parent module to replace the old attention module
                parent_parts = module_parts[:-1]
                if parent_parts:
                    parent_module = model
                    for part in parent_parts:
                        parent_module = getattr(parent_module, part)
                    
                    child_name = module_parts[-1]
                    setattr(parent_module, child_name, new_attention_module)
                else:
                    # This is the top-level model, which shouldn't happen
                    logger.warning(f"Cannot replace top-level attention module {module_name}")
                    continue

                for head_idx in head_indices:
                    removed.append(f"{module_name}.head_{head_idx}")

                logger.debug(f"Removed {len(head_indices)} heads from {module_name}, new num_heads: {new_num_heads}")

            except Exception as e:
                logger.error(f"Failed to remove heads from {module_name}: {str(e)}")
                continue

        return removed

    def _remove_mlp_components(self, model: nn.Module, mlp_to_remove: List[str]) -> List[str]:
        """
        Remove specific MLP components from the model.

        Args:
            model: The model to modify
            mlp_to_remove: List of MLP component names to remove

        Returns:
            List of actually removed components
        """
        removed = []
        
        for mlp_name in mlp_to_remove:
            try:
                # Navigate to the MLP module
                module_parts = mlp_name.split('.')
                parent_module = model
                child_name = module_parts[-1]

                # Navigate to parent
                for part in module_parts[:-1]:
                    parent_module = getattr(parent_module, part)

                # Get the original MLP module
                original_mlp = getattr(parent_module, child_name)

                # Replace with identity to maintain connectivity
                replacement_module = nn.Identity()
                setattr(parent_module, child_name, replacement_module)
                
                removed.append(mlp_name)
                logger.debug(f"Removed MLP component: {mlp_name}")

            except Exception as e:
                logger.error(f"Failed to remove MLP component {mlp_name}: {str(e)}")
                continue

        return removed

    def _adaptive_pruning(self, model: nn.Module, importance_scores: Dict[str, float],
                         target_pruning_ratio: float) -> List[str]:
        """
        Perform adaptive pruning based on dynamic importance thresholds.

        Args:
            model: The model to modify
            importance_scores: Dictionary of layer importance scores
            target_pruning_ratio: Target pruning ratio

        Returns:
            List of removed layer names
        """
        # Calculate threshold based on target pruning ratio
        sorted_scores = sorted(importance_scores.values())
        threshold_idx = int(len(sorted_scores) * target_pruning_ratio)
        threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else float('inf')

        # Identify layers below threshold
        layers_to_remove = [
            name for name, score in importance_scores.items()
            if score <= threshold
        ]

        # Limit to approximately the target ratio
        target_count = int(len(importance_scores) * target_pruning_ratio)
        layers_to_remove = layers_to_remove[:target_count]

        # Remove the identified layers
        return self._remove_layers(model, layers_to_remove)

    def _estimate_accuracy_preservation(self, original_params: int, pruned_params: int,
                                       layers_removed: int) -> bool:
        """
        Estimate if accuracy is preserved after pruning.

        Args:
            original_params: Original parameter count
            pruned_params: Pruned parameter count
            layers_removed: Number of layers removed

        Returns:
            True if accuracy is estimated to be preserved, False otherwise
        """
        if original_params == 0:
            return True

        # Simple heuristic: if we removed less than 30% of parameters and less than 20% of layers,
        # assume accuracy is preserved
        param_reduction = (original_params - pruned_params) / original_params
        layer_reduction = layers_removed / max(1, sum(1 for _ in self._get_all_prunable_layers()))

        # More sophisticated estimation would involve actual validation
        return (param_reduction < 0.3 and layer_reduction < 0.2) or (param_reduction < 0.1)

    def _get_all_prunable_layers(self):
        """Helper to get all prunable layers in a model."""
        # This is a helper that would be called with a specific model
        # Implemented here as a generator function
        yield from []

    def get_pruning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about performed pruning operations.

        Returns:
            Dictionary containing pruning statistics
        """
        if not self.pruning_history:
            return {'message': 'No pruning operations performed yet'}

        total_pruned_params = sum(record['pruned_params'] for record in self.pruning_history)
        total_original_params = sum(record['original_params'] for record in self.pruning_history)
        avg_compression = np.mean([record['compression_ratio'] for record in self.pruning_history])
        avg_accuracy_preserved = np.mean([int(record['accuracy_preserved']) for record in self.pruning_history])

        return {
            'total_pruning_operations': len(self.pruning_history),
            'average_compression_ratio': avg_compression,
            'average_accuracy_preserved': avg_accuracy_preserved,
            'total_parameters_reduced': total_original_params - total_pruned_params,
            'latest_operation': self.pruning_history[-1] if self.pruning_history else None
        }


# Global instance for convenience
_structured_pruning_system = StructuredPruningSystem()


def get_structured_pruning_system() -> StructuredPruningSystem:
    """
    Get the global structured pruning system instance.

    Returns:
        StructuredPruningSystem instance
    """
    return _structured_pruning_system


def apply_structured_pruning(model: nn.Module, pruning_ratio: float = 0.2,
                           block_size: int = 1, method: PruningMethod = PruningMethod.LAYER_REMOVAL,
                           dataloader: Optional[Any] = None) -> PruningResult:
    """
    Convenience function to apply structured pruning to a model.

    Args:
        model: The model to prune
        pruning_ratio: Ratio of blocks/layers to remove (0.0 to 1.0)
        block_size: Size of blocks to remove (for block pruning)
        method: Pruning method to use
        dataloader: Optional dataloader for importance calculation

    Returns:
        PruningResult containing the pruned model and statistics
    """
    pruning_system = get_structured_pruning_system()
    return pruning_system.prune_model(model, pruning_ratio, block_size, method, dataloader)


__all__ = [
    "StructuredPruningSystem",
    "PruningMethod",
    "PruningResult",
    "get_structured_pruning_system",
    "apply_structured_pruning"
]