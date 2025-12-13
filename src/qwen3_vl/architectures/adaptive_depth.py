"""
Input Complexity Assessment for Adaptive Depth Networks in Qwen3-VL Architecture
This module implements functionality to assess the complexity of inputs for both
vision and language modalities, enabling adaptive depth selection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from src.qwen3_vl.config import Qwen3VLConfig


class InputComplexityAssessor(nn.Module):
    """
    Assess the complexity of inputs for both vision and language modalities
    to enable adaptive depth selection in the Qwen3-VL architecture.
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size
        
        # Text complexity assessment components
        self.text_complexity_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Image complexity assessment components
        self.image_complexity_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.image_complexity_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Multimodal fusion complexity assessment
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + 32, self.hidden_size // 2),  # 32 from image conv features
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # For text complexity, we'll use token statistics
        self.token_freq_embed = nn.Embedding(config.vocab_size, 16)
        self.token_freq_projector = nn.Linear(16, 1)
        
        # Positional complexity assessment
        self.positional_complexity = nn.Parameter(torch.randn(512, 1) * 0.02)  # Max position complexity

    def assess_text_complexity(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Assess the complexity of text input based on multiple factors:
        - Token frequency patterns
        - Repetition patterns
        - Positional information
        """
        batch_size, seq_len = input_ids.shape
        
        # Calculate token frequency statistics
        unique_tokens = torch.unique(input_ids, return_counts=False)
        total_tokens = seq_len * batch_size
        unique_ratio = len(unique_tokens) / total_tokens
        
        # Calculate repetition patterns
        if seq_len > 1:
            # Count consecutive identical tokens
            consecutive_same = (input_ids[:, 1:] == input_ids[:, :-1]).float().mean()
        else:
            consecutive_same = torch.tensor(0.0, device=input_ids.device)
        
        # Use embedding-based complexity assessment
        token_embeddings = self.token_freq_embed(input_ids)
        token_complexity = self.token_freq_projector(token_embeddings).squeeze(-1)
        avg_token_complexity = token_complexity.mean()
        
        # Positional complexity
        if seq_len <= self.positional_complexity.size(0):
            pos_complexity = self.positional_complexity[:seq_len].mean()
        else:
            pos_complexity = self.positional_complexity.mean()
        
        # Combine all factors into a single complexity score
        # Normalize between 0 and 1
        complexity_score = torch.clamp(
            0.3 * unique_ratio + 
            0.2 * (1 - consecutive_same) +  # Less repetition = higher complexity
            0.3 * avg_token_complexity + 
            0.2 * torch.abs(pos_complexity),
            0.0, 1.0
        )
        
        return complexity_score

    def assess_image_complexity(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Assess the complexity of image input based on:
        - Texture information
        - Edge detection
        - Color variation
        """
        batch_size, channels, height, width = pixel_values.shape
        
        # Use convolutional features to assess complexity
        conv_features = self.image_complexity_conv(pixel_values)
        conv_features_flat = conv_features.view(batch_size, -1)
        
        # Calculate gradient-based complexity (edges)
        if height > 1 and width > 1:
            # Calculate gradients
            grad_x = torch.abs(pixel_values[:, :, :, 1:] - pixel_values[:, :, :, :-1])
            grad_y = torch.abs(pixel_values[:, :, 1:, :] - pixel_values[:, :, :-1, :])
            
            # Average gradient magnitude
            avg_grad_x = grad_x.mean()
            avg_grad_y = grad_y.mean()
            edge_complexity = (avg_grad_x + avg_grad_y) / 2
        else:
            edge_complexity = torch.tensor(0.0, device=pixel_values.device)
        
        # Calculate color variation
        pixel_mean = pixel_values.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        color_variation = torch.abs(pixel_values - pixel_mean).mean()
        
        # Get complexity from convolutional features
        image_complexity_score = self.image_complexity_head(conv_features_flat).mean()
        
        # Combine all factors
        combined_complexity = torch.clamp(
            0.4 * image_complexity_score + 
            0.3 * edge_complexity + 
            0.3 * color_variation,
            0.0, 1.0
        )
        
        return combined_complexity

    def assess_multimodal_complexity(self, input_ids: torch.Tensor, 
                                   pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Assess the complexity of multimodal input combining both text and image.
        """
        text_complexity = self.assess_text_complexity(input_ids)
        image_complexity = self.assess_image_complexity(pixel_values)
        
        # Extract features for multimodal fusion
        # Use a simplified approach to get text features
        text_features = torch.zeros(input_ids.size(0), self.hidden_size, device=input_ids.device)
        # Use average embedding as a simple representation
        if hasattr(self, 'dummy_text_embed'):
            text_embeds = self.dummy_text_embed(input_ids)
            text_features = text_embeds.mean(dim=1)  # Average across sequence
        else:
            # Create a simple embedding for complexity assessment
            dummy_embed = nn.Embedding(self.config.vocab_size, self.hidden_size).to(input_ids.device)
            text_embeds = dummy_embed(input_ids)
            text_features = text_embeds.mean(dim=1)  # Average across sequence
        
        # Get image conv features for fusion
        image_conv_features = self.image_complexity_conv(pixel_values)
        image_conv_flat = image_conv_features.view(input_ids.size(0), -1)
        
        # Combine text and image features for multimodal complexity
        combined_features = torch.cat([text_features, image_conv_flat], dim=1)
        multimodal_complexity = self.multimodal_fusion(combined_features).mean()
        
        # Combine all complexity measures
        final_complexity = torch.clamp(
            0.3 * text_complexity + 
            0.3 * image_complexity + 
            0.4 * multimodal_complexity,
            0.0, 1.0
        )
        
        return final_complexity

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to assess input complexity.
        Returns a complexity score between 0 and 1.
        """
        if input_ids is not None and pixel_values is not None:
            # Multimodal input
            return self.assess_multimodal_complexity(input_ids, pixel_values)
        elif input_ids is not None:
            # Text-only input
            return self.assess_text_complexity(input_ids)
        elif pixel_values is not None:
            # Image-only input
            return self.assess_image_complexity(pixel_values)
        else:
            # No input provided
            return torch.tensor(0.5, device=self.positional_complexity.device)  # Default medium complexity


class AdaptiveDepthController(nn.Module):
    """
    Controller that uses complexity assessment to determine how many layers to use.
    """
    def __init__(self, config: Qwen3VLConfig, complexity_assessor: InputComplexityAssessor):
        super().__init__()
        self.config = config
        self.complexity_assessor = complexity_assessor
        
        # Parameters to map complexity to depth
        self.complexity_to_depth = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Minimum and maximum depth allowed
        self.min_depth_ratio = getattr(config, 'min_depth_ratio', 0.2)  # At least 20% of layers
        self.max_depth_ratio = getattr(config, 'max_depth_ratio', 1.0)  # Up to 100% of layers
        
        # Temperature for soft depth selection
        self.temperature = getattr(config, 'depth_temperature', 1.0)

    def calculate_target_depth(self, complexity_score: torch.Tensor) -> torch.Tensor:
        """
        Calculate the target depth based on complexity score.
        """
        # Map complexity to depth ratio using the neural network
        depth_ratio = self.complexity_to_depth(complexity_score.unsqueeze(0) if complexity_score.dim() == 0 else complexity_score)
        
        # Clamp to valid range
        depth_ratio = torch.clamp(depth_ratio, self.min_depth_ratio, self.max_depth_ratio)
        
        # Calculate actual number of layers to use
        target_depth = depth_ratio * self.config.num_hidden_layers
        
        return target_depth, depth_ratio

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None) -> Tuple[int, float]:
        """
        Forward pass to determine the number of layers to use based on input complexity.
        
        Returns:
            - int: Number of layers to use
            - float: Complexity score (0-1)
        """
        # Assess complexity
        complexity_score = self.complexity_assessor(input_ids, pixel_values)
        
        # Calculate target depth
        target_depth, depth_ratio = self.calculate_target_depth(complexity_score)
        
        # Convert to integer number of layers (round to nearest)
        num_layers_to_use = int(torch.round(target_depth).item())
        
        # Ensure at least 1 layer is used
        num_layers_to_use = max(1, min(num_layers_to_use, self.config.num_hidden_layers))
        
        return num_layers_to_use, complexity_score.item()


def create_complexity_guided_mask(seq_len: int, complexity_score: float, 
                                device: torch.device) -> torch.Tensor:
    """
    Create a complexity-guided attention mask that emphasizes important positions
    based on the input complexity.
    """
    # Create a base mask that gradually increases attention with complexity
    base_mask = torch.ones(seq_len, seq_len, device=device) * complexity_score
    base_mask = torch.tril(base_mask)  # Maintain causality if needed
    
    # Add some structure based on complexity
    if complexity_score > 0.7:  # High complexity
        # More focused attention patterns
        diag_mask = torch.eye(seq_len, device=device) * 0.5
        base_mask = base_mask + diag_mask
    elif complexity_score < 0.3:  # Low complexity
        # More uniform attention
        uniform_mask = torch.ones(seq_len, seq_len, device=device) * 0.2
        base_mask = (base_mask + uniform_mask) / 2
    
    return torch.clamp(base_mask, 0.0, 1.0)