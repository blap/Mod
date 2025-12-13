"""
Model Pruning Optimization for Qwen3-VL Model
Implementation of structured and unstructured pruning techniques for CPU optimization on Intel i5-10210U

This module implements various pruning techniques to reduce model complexity and improve
inference speed on CPU, specifically targeting the Intel i5-10210U architecture.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import numpy as np
import logging
from collections import OrderedDict
import math


@dataclass
class PruningConfig:
    """Configuration for model pruning optimization."""
    # Pruning parameters
    pruning_method: str = "structured"  # 'structured', 'unstructured', or 'magnitude'
    pruning_ratio: float = 0.2  # Ratio of parameters to prune (0.0-1.0)
    structured_pruning_dim: str = "channel"  # 'channel', 'row', 'column' for structured pruning
    pruning_scope: str = "local"  # 'local' (per layer) or 'global' (across all layers)
    
    # Pruning schedule
    pruning_schedule: str = "iterative"  # 'one_shot', 'iterative', or 'gradual'
    num_pruning_steps: int = 10  # Number of pruning steps for iterative pruning
    initial_sparsity: float = 0.0  # Initial sparsity before pruning starts
    
    # Fine-tuning after pruning
    enable_fine_tuning: bool = True  # Whether to fine-tune after pruning
    fine_tune_epochs: int = 3  # Number of epochs for fine-tuning
    fine_tune_learning_rate: float = 1e-5  # Learning rate for fine-tuning
    
    # Performance optimization
    enable_pattern_pruning: bool = False  # Use pattern-based pruning (e.g., N:M patterns)
    pattern_ratio: Tuple[int, int] = (2, 4)  # N:M pattern (N zeros in every M weights)
    
    # Model-specific parameters
    prune_embeddings: bool = False  # Whether to prune embedding layers
    prune_attention: bool = True   # Whether to prune attention components
    prune_mlp: bool = True         # Whether to prune MLP components
    prune_output_layers: bool = False  # Whether to prune output layers


class StructuredPruning(nn.Module):
    """
    Module that applies structured pruning to neural network layers.
    """
    def __init__(self, config: PruningConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

    def apply_pruning_to_layer(
        self,
        layer: nn.Module,
        pruning_ratio: float,
        layer_name: str = ""
    ) -> nn.Module:
        """
        Apply structured pruning to a single layer.
        
        Args:
            layer: The layer to prune
            pruning_ratio: The ratio of parameters to prune
            layer_name: Name of the layer (for logging purposes)
            
        Returns:
            Pruned layer
        """
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if self.config.structured_pruning_dim == "channel":
                # For structured pruning by channel, we use a custom approach
                # as PyTorch's built-in pruning is mainly for unstructured pruning
                return self._apply_structured_channel_pruning(layer, pruning_ratio, layer_name)
            elif self.config.structured_pruning_dim == "row":
                # Prune entire rows
                prune.ln_structured(layer, name='weight', amount=pruning_ratio, n=1, dim=0)
            elif self.config.structured_pruning_dim == "column":
                # Prune entire columns
                prune.ln_structured(layer, name='weight', amount=pruning_ratio, n=1, dim=1)
        elif isinstance(layer, nn.Conv2d):
            # For convolutional layers, we can prune along different dimensions
            if self.config.structured_pruning_dim == "channel":
                # Prune output channels
                prune.ln_structured(layer, name='weight', amount=pruning_ratio, n=1, dim=0)
            elif self.config.structured_pruning_dim == "row":
                # Prune along spatial dimensions (height)
                # This requires a custom implementation
                return self._apply_structured_conv_pruning(layer, pruning_ratio, layer_name)
            elif self.config.structured_pruning_dim == "column":
                # Prune along spatial dimensions (width)
                # This requires a custom implementation
                return self._apply_structured_conv_pruning(layer, pruning_ratio, layer_name)
        else:
            # For other layer types, we can still apply unstructured pruning
            if self.config.pruning_method == "unstructured":
                prune.l1_unstructured(layer, name='weight', amount=pruning_ratio)
            elif self.config.pruning_method == "magnitude":
                prune.random_unstructured(layer, name='weight', amount=pruning_ratio)
        
        return layer

    def _apply_structured_channel_pruning(
        self,
        layer: nn.Linear,
        pruning_ratio: float,
        layer_name: str
    ) -> nn.Module:
        """
        Apply structured pruning by removing entire channels (output dimensions).
        This is a simplified implementation - a full implementation would require
        more complex reconfiguration of the network.
        """
        # Calculate number of channels to keep
        total_channels = layer.out_features
        channels_to_keep = int(total_channels * (1 - pruning_ratio))
        channels_to_keep = max(1, channels_to_keep)  # Keep at least one channel
        
        # For structured channel pruning, we'll use magnitude-based selection
        # to identify which channels to keep
        weight_norms = torch.norm(layer.weight, dim=1)  # L2 norm of each output channel
        _, indices_to_keep = torch.topk(weight_norms, channels_to_keep, largest=True)
        
        # Create a mask for the weights
        mask = torch.zeros_like(layer.weight)
        mask[indices_to_keep, :] = 1.0
        
        # Apply the mask (this is equivalent to pruning)
        layer.weight.data = layer.weight.data * mask
        
        # Log pruning information
        self.logger.info(f"Structured channel pruning applied to {layer_name}: "
                        f"Kept {channels_to_keep}/{total_channels} channels "
                        f"({pruning_ratio*100:.1f}% pruned)")
        
        return layer

    def _apply_structured_conv_pruning(
        self,
        layer: nn.Conv2d,
        pruning_ratio: float,
        layer_name: str
    ) -> nn.Module:
        """
        Apply structured pruning to convolutional layers.
        """
        # Calculate number of channels to keep
        total_channels = layer.out_channels
        channels_to_keep = int(total_channels * (1 - pruning_ratio))
        channels_to_keep = max(1, channels_to_keep)  # Keep at least one channel
        
        # For structured channel pruning in conv layers
        weight_norms = torch.norm(layer.weight, dim=(1, 2, 3))  # L2 norm of each output channel
        _, indices_to_keep = torch.topk(weight_norms, channels_to_keep, largest=True)
        
        # Create a mask for the weights
        mask = torch.zeros_like(layer.weight)
        mask[indices_to_keep, :, :, :] = 1.0
        
        # Apply the mask
        layer.weight.data = layer.weight.data * mask
        
        # Log pruning information
        self.logger.info(f"Structured conv pruning applied to {layer_name}: "
                        f"Kept {channels_to_keep}/{total_channels} channels "
                        f"({pruning_ratio*100:.1f}% pruned)")
        
        return layer


class PatternPruning(nn.Module):
    """
    Module that applies pattern-based pruning (e.g., N:M patterns).
    """
    def __init__(self, config: PruningConfig):
        super().__init__()
        self.config = config
        self.pattern_n, self.pattern_m = config.pattern_ratio
        self.logger = logging.getLogger(__name__)

    def apply_pattern_pruning_to_layer(
        self,
        layer: nn.Module,
        layer_name: str = ""
    ) -> nn.Module:
        """
        Apply pattern-based pruning to a single layer.
        For example, in 2:4 pattern pruning, 2 out of every 4 weights are set to zero.
        """
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            return layer

        # Get the weight tensor
        weight = layer.weight.data
        
        # Reshape weights to apply pattern pruning
        # For 2D weights (Linear layers): [out_features, in_features]
        if weight.dim() == 2:
            out_features, in_features = weight.shape
            
            # Reshape to groups of M weights
            if in_features % self.pattern_m != 0:
                # Pad with zeros if not divisible
                padding_size = self.pattern_m - (in_features % self.pattern_m)
                weight_padded = torch.nn.functional.pad(weight, (0, padding_size))
                reshaped = weight_padded.view(out_features, -1, self.pattern_m)
            else:
                reshaped = weight.view(out_features, -1, self.pattern_m)
            
            # Apply N:M pattern - keep N largest weights in each group of M
            values, indices = torch.topk(reshaped.abs(), self.pattern_n, dim=-1, largest=True)
            
            # Create mask
            mask = torch.zeros_like(reshaped)
            mask.scatter_(-1, indices, 1.0)
            
            # Apply mask
            reshaped_pruned = reshaped * mask
            
            # Reshape back to original shape
            pruned_weight = reshaped_pruned.view(out_features, -1)
            if padding_size > 0:
                pruned_weight = pruned_weight[:, :-padding_size]
            
            layer.weight.data = pruned_weight
        elif weight.dim() == 4:  # Conv2d weights: [out_channels, in_channels, H, W]
            out_channels, in_channels, h, w = weight.shape
            total_elements = in_channels * h * w
            
            # Reshape to apply pattern along the channel dimension
            reshaped = weight.view(out_channels, -1)  # [out_channels, in_channels*H*W]
            
            # Apply pattern pruning
            if total_elements % self.pattern_m != 0:
                # Pad with zeros if not divisible
                padding_size = self.pattern_m - (total_elements % self.pattern_m)
                reshaped_padded = torch.nn.functional.pad(reshaped, (0, padding_size))
                reshaped_groups = reshaped_padded.view(out_channels, -1, self.pattern_m)
            else:
                reshaped_groups = reshaped.view(out_channels, -1, self.pattern_m)
            
            # Apply N:M pattern
            values, indices = torch.topk(reshaped_groups.abs(), self.pattern_n, dim=-1, largest=True)
            
            # Create mask
            mask = torch.zeros_like(reshaped_groups)
            mask.scatter_(-1, indices, 1.0)
            
            # Apply mask
            reshaped_pruned = reshaped_groups * mask
            
            # Reshape back to original shape
            pruned_weight = reshaped_pruned.view(out_channels, -1)
            if padding_size > 0:
                pruned_weight = pruned_weight[:, :-padding_size]
            
            layer.weight.data = pruned_weight.view(out_channels, in_channels, h, w)

        # Log pruning information
        total_params = weight.numel()
        pruned_params = (layer.weight.data == 0).sum().item()
        prune_ratio = pruned_params / total_params if total_params > 0 else 0
        
        self.logger.info(f"Pattern pruning applied to {layer_name}: "
                        f"{prune_ratio*100:.1f}% of weights pruned "
                        f"with {self.pattern_n}:{self.pattern_m} pattern")
        
        return layer


class ModelPruner:
    """
    Main class for applying pruning to the Qwen3-VL model.
    """
    def __init__(self, pruning_config: PruningConfig = None):
        self.config = pruning_config or PruningConfig()
        self.logger = logging.getLogger(__name__)
        self.structured_pruner = StructuredPruning(self.config)
        self.pattern_pruner = PatternPruning(self.config) if self.config.enable_pattern_pruning else None

    def apply_pruning_to_model(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply pruning to the Qwen3-VL model.
        
        Args:
            model: The Qwen3-VL model to prune
            
        Returns:
            Tuple of (pruned_model, pruning_info)
        """
        self.logger.info("Applying pruning to the model...")
        
        # Identify layers to prune based on configuration
        layers_to_prune = self._identify_layers_to_prune(model)
        
        # Apply pruning based on schedule
        if self.config.pruning_schedule == "one_shot":
            self._apply_one_shot_pruning(model, layers_to_prune)
        elif self.config.pruning_schedule == "iterative":
            self._apply_iterative_pruning(model, layers_to_prune)
        elif self.config.pruning_schedule == "gradual":
            self._apply_gradual_pruning(model, layers_to_prune)
        
        # Create pruning info
        pruning_info = {
            'config': self.config,
            'pruning_method': self.config.pruning_method,
            'pruning_ratio': self.config.pruning_ratio,
            'pruning_schedule': self.config.pruning_schedule,
            'prune_embeddings': self.config.prune_embeddings,
            'prune_attention': self.config.prune_attention,
            'prune_mlp': self.config.prune_mlp,
            'prune_output_layers': self.config.prune_output_layers
        }
        
        self.logger.info("Model pruning applied successfully!")
        return model, pruning_info

    def _identify_layers_to_prune(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Identify which layers should be pruned based on the configuration.
        """
        layers_to_prune = []
        
        for name, module in model.named_modules():
            # Check if this module should be pruned based on name and type
            should_prune = False
            
            # Prune attention components if specified
            if self.config.prune_attention and any(attn_type in name.lower() for attn_type in ['attn', 'attention']):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    should_prune = True
            
            # Prune MLP components if specified
            elif self.config.prune_mlp and any(mlp_type in name.lower() for mlp_type in ['mlp', 'ffn', 'linear']):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Avoid pruning output layers if specified
                    if not (self.config.prune_output_layers == False and 'output' in name.lower()):
                        should_prune = True
            
            # Prune embeddings if specified
            elif self.config.prune_embeddings and any(emb_type in name.lower() for emb_type in ['embed', 'embedding']):
                if isinstance(module, (nn.Embedding, nn.Linear)):
                    should_prune = True
            
            # Prune output layers if specified
            elif self.config.prune_output_layers and any(out_type in name.lower() for out_type in ['output', 'classifier', 'lm_head']):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    should_prune = True
            
            # Add to list if it's a prunable layer type
            if should_prune and isinstance(module, (nn.Linear, nn.Conv2d)):
                layers_to_prune.append((name, module))
        
        self.logger.info(f"Identified {len(layers_to_prune)} layers for pruning")
        return layers_to_prune

    def _apply_one_shot_pruning(self, model: nn.Module, layers_to_prune: List[Tuple[str, nn.Module]]):
        """
        Apply one-shot pruning (prune all at once).
        """
        self.logger.info("Applying one-shot pruning...")
        
        for name, layer in layers_to_prune:
            if self.config.enable_pattern_pruning and self.pattern_pruner:
                self.pattern_pruner.apply_pattern_pruning_to_layer(layer, name)
            else:
                self.structured_pruner.apply_pruning_to_layer(layer, self.config.pruning_ratio, name)

    def _apply_iterative_pruning(self, model: nn.Module, layers_to_prune: List[Tuple[str, nn.Module]]):
        """
        Apply iterative pruning (gradually increase sparsity).
        """
        self.logger.info(f"Applying iterative pruning over {self.config.num_pruning_steps} steps...")
        
        # Calculate pruning increment
        target_ratio = self.config.pruning_ratio
        initial_ratio = self.config.initial_sparsity
        step_size = (target_ratio - initial_ratio) / self.config.num_pruning_steps
        
        for step in range(self.config.num_pruning_steps):
            current_ratio = initial_ratio + (step + 1) * step_size
            
            for name, layer in layers_to_prune:
                if self.config.enable_pattern_pruning and self.pattern_pruner:
                    # For iterative pattern pruning, we'll reapply the pattern at each step
                    # This is a simplified approach - in practice, you might want to fine-tune between steps
                    self.pattern_pruner.apply_pattern_pruning_to_layer(layer, name)
                else:
                    # Reset pruning before applying new level
                    if hasattr(layer, 'weight_mask'):
                        # Remove existing pruning
                        try:
                            prune.remove(layer, 'weight')
                        except ValueError:
                            pass  # Not pruned yet
                    
                    # Apply new pruning level
                    self.structured_pruner.apply_pruning_to_layer(layer, current_ratio, name)
            
            self.logger.info(f"Iterative pruning step {step + 1}/{self.config.num_pruning_steps} "
                            f"completed (sparsity: {current_ratio:.2f})")

    def _apply_gradual_pruning(self, model: nn.Module, layers_to_prune: List[Tuple[str, nn.Module]]):
        """
        Apply gradual pruning (slowly increase sparsity during training).
        This method is meant to be used during the training process.
        """
        self.logger.info("Applying gradual pruning...")
        
        # For gradual pruning, we'll set up the layers but the actual pruning
        # happens during training iterations
        for name, layer in layers_to_prune:
            # For now, we'll just prepare the layers for gradual pruning
            # In a full implementation, this would involve more complex logic
            # to gradually increase sparsity during training
            if self.config.enable_pattern_pruning and self.pattern_pruner:
                self.pattern_pruner.apply_pattern_pruning_to_layer(layer, name)
            else:
                # Apply initial sparsity
                self.structured_pruner.apply_pruning_to_layer(layer, self.config.initial_sparsity, name)

    def fine_tune_model(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        device: torch.device
    ) -> nn.Module:
        """
        Fine-tune the pruned model to recover accuracy.
        
        Args:
            model: The pruned model to fine-tune
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to run training on
            
        Returns:
            Fine-tuned model
        """
        if not self.config.enable_fine_tuning:
            self.logger.info("Fine-tuning is disabled, skipping...")
            return model

        self.logger.info(f"Starting fine-tuning for {self.config.fine_tune_epochs} epochs...")
        
        # Set model to train mode
        model.train()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.fine_tune_learning_rate,
            weight_decay=0.01
        )
        
        # Setup loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config.fine_tune_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    if isinstance(batch, dict):
                        outputs = model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in batch.items()})
                    elif isinstance(batch, (list, tuple)):
                        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                        outputs = model(*batch)
                    else:
                        outputs = model(batch.to(device))
                    
                    # Calculate loss (this assumes a specific output format)
                    # In practice, you'd need to adjust this based on your model's output
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        # If no loss is provided, we can't proceed with training
                        # This is a simplified example
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update parameters
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error in fine-tuning batch: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.logger.info(f"Fine-tuning epoch {epoch + 1}/{self.config.fine_tune_epochs}, "
                            f"avg loss: {avg_loss:.4f}")
        
        # Set model back to eval mode
        model.eval()
        
        self.logger.info("Fine-tuning completed!")
        return model

    def calculate_pruning_metrics(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module
    ) -> Dict[str, float]:
        """
        Calculate metrics for the pruning effect.
        
        Args:
            original_model: Original model before pruning
            pruned_model: Model after pruning
            
        Returns:
            Dictionary with pruning metrics
        """
        # Count parameters before and after pruning
        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        # Count zero parameters in pruned model
        zero_params = sum((p == 0).sum().item() for p in pruned_model.parameters())
        
        # Calculate metrics
        parameter_reduction = (original_params - pruned_params) / original_params * 100
        sparsity_ratio = zero_params / pruned_params * 100 if pruned_params > 0 else 0
        compression_ratio = original_params / pruned_params if pruned_params > 0 else float('inf')
        
        return {
            'original_parameter_count': original_params,
            'pruned_parameter_count': pruned_params,
            'zero_parameter_count': zero_params,
            'parameter_reduction_percent': parameter_reduction,
            'sparsity_ratio_percent': sparsity_ratio,
            'compression_ratio': compression_ratio
        }

    def benchmark_pruning_impact(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        test_data_loader
    ) -> Dict[str, Any]:
        """
        Benchmark the impact of pruning on model performance.
        
        Args:
            original_model: The original model
            pruned_model: The pruned model
            test_data_loader: DataLoader with test data
            
        Returns:
            Dictionary with performance metrics
        """
        # Set both models to evaluation mode
        original_model.eval()
        pruned_model.eval()
        
        # Track performance metrics
        original_times = []
        pruned_times = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                if i >= 10:  # Limit to 10 batches for quick benchmarking
                    break
                
                # Benchmark original model
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                else:
                    import time
                    start_time_cpu = time.time()
                
                try:
                    if isinstance(batch, dict):
                        _ = original_model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = original_model(*batch)
                    else:
                        _ = original_model(batch)
                except Exception as e:
                    self.logger.warning(f"Original model benchmark failed for batch {i}: {e}")
                    continue
                
                if start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    original_time = start_time.elapsed_time(end_time)
                else:
                    end_time_cpu = time.time()
                    original_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to milliseconds
                
                original_times.append(original_time)
                
                # Benchmark pruned model
                if start_time:
                    start_time.record()
                
                try:
                    if isinstance(batch, dict):
                        _ = pruned_model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = pruned_model(*batch)
                    else:
                        _ = pruned_model(batch)
                except Exception as e:
                    self.logger.warning(f"Pruned model benchmark failed for batch {i}: {e}")
                    continue
                
                if start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    pruned_time = start_time.elapsed_time(end_time)
                else:
                    start_time_cpu = time.time()
                    _ = pruned_model(batch)
                    end_time_cpu = time.time()
                    pruned_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to milliseconds
                
                pruned_times.append(pruned_time)
        
        # Calculate metrics
        avg_original_time = np.mean(original_times) if original_times else 0
        avg_pruned_time = np.mean(pruned_times) if pruned_times else 0
        speedup = avg_original_time / avg_pruned_time if avg_pruned_time > 0 else float('inf')
        
        # Calculate model sizes
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024**2)  # MB
        pruned_size = sum(p.numel() * p.element_size() for p in pruned_model.parameters()) / (1024**2)  # MB
        size_reduction = (original_size - pruned_size) / original_size * 100 if original_size > 0 else 0
        
        return {
            'original_avg_time_ms': avg_original_time,
            'pruned_avg_time_ms': avg_pruned_time,
            'speedup': speedup,
            'original_model_size_mb': original_size,
            'pruned_model_size_mb': pruned_size,
            'size_reduction_percent': size_reduction,
            'num_test_batches': len(original_times)
        }


def apply_pruning_to_model(
    model: nn.Module,
    config: Optional[PruningConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply pruning to the Qwen3-VL model.

    Args:
        model: The Qwen3-VL model to prune
        config: Configuration for pruning (optional)

    Returns:
        Tuple of (pruned_model, pruning_info)
    """
    logger = logging.getLogger(__name__)
    logger.info("Applying pruning to the Qwen3-VL model...")

    # Use default config if none provided
    if config is None:
        config = PruningConfig()

    # Initialize the pruner
    pruner = ModelPruner(config)

    # Apply pruning
    pruned_model, pruning_info = pruner.apply_pruning_to_model(model)

    logger.info("Model pruning applied successfully!")
    return pruned_model, pruning_info


if __name__ == "__main__":
    print("Model Pruning Optimization for Qwen3-VL Model")
    print("=" * 60)
    print("This module implements structured and unstructured pruning techniques")
    print("for CPU optimization targeting Intel i5-10210U architecture")
    print("=" * 60)
    
    # Example usage
    config = PruningConfig()
    print(f"Default pruning config: {config}")