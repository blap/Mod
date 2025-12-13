"""
Conditional feature extraction based on input modality requirements
for the Qwen3-VL architecture.

This module implements conditional feature extraction that activates
different pathways based on input modality, with modality-specific
feature extraction mechanisms to optimize processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
from qwen3_vl.config.config import Qwen3VLConfig


class TextFeatureExtractor(nn.Module):
    """Text-specific feature extractor optimized for language processing."""
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Optional: Add layer normalization for better feature normalization
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract features from text input.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Extracted text features of shape (batch_size, seq_len, hidden_size)
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply normalization and dropout
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class VisionFeatureExtractor(nn.Module):
    """Vision-specific feature extractor optimized for image processing."""
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            bias=False
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Positional encoding for vision features
        self.num_patches_per_dim = config.vision_image_size // config.vision_patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        # Learnable positional embeddings
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image input.
        
        Args:
            pixel_values: Input pixel values of shape (batch_size, channels, height, width)
            
        Returns:
            Extracted vision features of shape (batch_size, num_patches, hidden_size)
        """
        batch_size = pixel_values.shape[0]
        image_height, image_width = pixel_values.shape[-2], pixel_values.shape[-1]
        
        # Calculate how many patches we'll have based on input image size
        patch_height = image_height // self.config.vision_patch_size
        patch_width = image_width // self.config.vision_patch_size
        num_patches = patch_height * patch_width
        
        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values)  # shape (batch_size, embed_dim, patch_height, patch_width)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # shape (batch_size, num_patches, embed_dim)
        
        # Add positional embeddings
        if num_patches <= self.num_patches:
            # Use the first num_patches position embeddings
            position_embeddings = self.position_embedding.weight[:num_patches].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # If more patches than expected, interpolate positional embeddings
            position_embeddings = F.interpolate(
                self.position_embedding.weight.unsqueeze(0).transpose(1, 2),
                size=num_patches,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1).expand(batch_size, -1, -1)
        
        hidden_states = patch_embeds + position_embeddings
        
        # Apply normalization and dropout
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MultimodalFusion(nn.Module):
    """Fusion module for combining text and vision features."""
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Projection layers for aligning feature dimensions if needed
        self.text_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.vision_projection = nn.Linear(config.vision_hidden_size, config.hidden_size, bias=False)
        
        # Cross-attention mechanism for fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=min(8, config.num_attention_heads // 4),  # Use fewer heads for fusion
            dropout=config.attention_dropout_prob,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self, 
        text_features: torch.Tensor, 
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and vision features.
        
        Args:
            text_features: Text features of shape (batch_size, text_seq_len, hidden_size)
            vision_features: Vision features of shape (batch_size, vision_seq_len, hidden_size)
            
        Returns:
            Fused features of shape (batch_size, combined_seq_len, hidden_size)
        """
        # Project features to common dimension if needed
        text_proj = self.text_projection(text_features)
        vision_proj = self.vision_projection(vision_features)

        # Concatenate features
        combined_features = torch.cat([vision_proj, text_proj], dim=1)

        # Apply cross-attention for fusion
        fused_features, _ = self.fusion_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features
        )

        # Apply normalization and dropout
        fused_features = self.norm(fused_features)
        fused_features = self.dropout(fused_features)

        return fused_features


class ModalityClassifier(nn.Module):
    """Classifies input modality to determine which extraction pathway to use."""
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Simple classification based on input presence
        # In practice, this could be more sophisticated
        self.modality_embedding = nn.Embedding(4, config.hidden_size)  # 4 modalities: text, vision, multimodal, unknown
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, 3)  # text, vision, multimodal
        self.activation = nn.Softmax(dim=-1)
    
    def forward(
        self, 
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None
    ) -> Tuple[str, torch.Tensor]:
        """
        Classify the input modality.
        
        Args:
            text_input: Text input tensor or None
            image_input: Image input tensor or None
            
        Returns:
            Tuple of (modality string, confidence scores)
        """
        # Determine modality based on input presence
        has_text = text_input is not None
        has_image = image_input is not None
        
        if has_text and has_image:
            modality_idx = 2  # multimodal
            modality = "multimodal"
        elif has_text:
            modality_idx = 0  # text
            modality = "text"
        elif has_image:
            modality_idx = 1  # vision
            modality = "vision"
        else:
            modality_idx = 3  # unknown
            modality = "unknown"
        
        # Create embedding for the determined modality
        modality_emb = self.modality_embedding(torch.tensor([modality_idx], device=text_input.device if has_text else image_input.device))
        
        # Get classification scores
        scores = self.classifier(modality_emb)
        confidence_scores = self.activation(scores)

        return modality, confidence_scores


class ComplexityAssessor(nn.Module):
    """Assesses input complexity to optimize feature extraction depth."""
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # For text complexity: analyze token patterns
        self.text_complexity_head = nn.Linear(config.hidden_size, 1)
        
        # For vision complexity: analyze pixel variance
        self.vision_complexity_head = nn.Linear(config.vision_hidden_size, 1)

        # Sigmoid to get complexity score between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self, 
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None
    ) -> float:
        """
        Assess the complexity of the input.
        
        Args:
            text_input: Original text input or None
            image_input: Original image input or None
            text_features: Processed text features or None
            vision_features: Processed vision features or None
            
        Returns:
            Complexity score between 0 and 1
        """
        if text_input is not None:
            # Assess text complexity based on token diversity
            # For simplicity, we'll use a basic measure of token repetition
            unique_tokens = torch.unique(text_input).size(0)
            total_tokens = text_input.numel()
            diversity_score = unique_tokens / total_tokens
            
            # Use the processed features if available for more sophisticated analysis
            if text_features is not None:
                # Calculate variance in the features as a measure of complexity
                feature_variance = torch.var(text_features, dim=-1).mean().item()
                # Normalize to 0-1 range based on expected variance range
                normalized_variance = min(feature_variance / 10.0, 1.0)  # Heuristic normalization
                
                # Combine diversity and variance measures
                complexity_score = 0.5 * diversity_score + 0.5 * normalized_variance
            else:
                complexity_score = diversity_score
            
        elif image_input is not None:
            # Assess vision complexity based on pixel variance
            pixel_variance = torch.var(image_input, dim=(1, 2, 3)).mean().item()
            # Normalize to 0-1 range based on expected variance range
            complexity_score = min(pixel_variance * 10.0, 1.0)  # Heuristic normalization
            
            # Use processed features if available for more sophisticated analysis
            if vision_features is not None:
                feature_variance = torch.var(vision_features, dim=-1).mean().item()
                normalized_variance = min(feature_variance / 5.0, 1.0)  # Heuristic normalization
                complexity_score = 0.7 * complexity_score + 0.3 * normalized_variance
        else:
            # No input provided, return neutral complexity
            complexity_score = 0.5
        
        return min(max(complexity_score, 0.0), 1.0)  # Clamp between 0 and 1


class ModalitySpecificExtractor(nn.Module):
    """Container for modality-specific feature extractors."""
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Initialize modality-specific extractors
        self.text_extractor = TextFeatureExtractor(config)
        self.vision_extractor = VisionFeatureExtractor(config)
        self.multimodal_fusion = MultimodalFusion(config)
    
    def extract_text_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract features from text input."""
        return self.text_extractor(input_ids)
    
    def extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from vision input."""
        return self.vision_extractor(pixel_values)
    
    def fuse_multimodal_features(
        self, 
        text_features: torch.Tensor, 
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse text and vision features."""
        return self.multimodal_fusion(text_features, vision_features)


class ConditionalFeatureExtractor(nn.Module):
    """
    Main conditional feature extraction module that activates different pathways
    based on input modality requirements.
    
    This module maintains full capacity with 32 transformer layers and 32 attention heads
    while optimizing processing through conditional pathways.
    """
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Initialize modality-specific extractor
        self.modality_specific_extractor = ModalitySpecificExtractor(config)
        
        # Initialize modality classifier
        self.modality_classifier = ModalityClassifier(config)

        # Initialize complexity assessor
        self.complexity_assessor = ComplexityAssessor(config)
    
    def forward(
        self,
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Union[str, float, torch.Tensor]]]:
        """
        Perform conditional feature extraction based on input modality.

        Args:
            text_input: Input token IDs of shape (batch_size, seq_len) or None
            image_input: Input pixel values of shape (batch_size, channels, height, width) or None
            attention_mask: Attention mask for text input or None
            **kwargs: Additional arguments

        Returns:
            Tuple of (extracted_features, modality_info_dict)
            - extracted_features: Features of shape (batch_size, seq_len, hidden_size)
            - modality_info_dict: Dictionary containing modality information
        """
        if text_input is None and image_input is None:
            raise ValueError("At least one of text_input or image_input must be provided")

        # Validate input tensor shapes
        if text_input is not None:
            if text_input.dim() != 2:
                raise ValueError(f"text_input must be 2D tensor, got {text_input.dim()}D")
            if text_input.size(0) == 0 or text_input.size(1) == 0:
                raise ValueError("text_input has empty dimensions")

        if image_input is not None:
            if image_input.dim() != 4:
                raise ValueError(f"image_input must be 4D tensor (batch, channels, height, width), got {image_input.dim()}D")
            if image_input.size(0) == 0:
                raise ValueError("image_input has empty batch dimension")

        # Classify input modality
        modality, confidence_scores = self.modality_classifier(text_input, image_input)

        # Assess input complexity
        complexity_score = self.complexity_assessor(text_input, image_input)

        # Extract features based on modality
        if modality == "text":
            if text_input is None:
                raise ValueError("Text input required for text modality")
            features = self.modality_specific_extractor.extract_text_features(text_input)

        elif modality == "vision":
            if image_input is None:
                raise ValueError("Image input required for vision modality")
            features = self.modality_specific_extractor.extract_vision_features(image_input)

        elif modality == "multimodal":
            if text_input is None or image_input is None:
                raise ValueError("Both text and image inputs required for multimodal processing")

            # Extract features separately
            text_features = self.modality_specific_extractor.extract_text_features(text_input)
            vision_features = self.modality_specific_extractor.extract_vision_features(image_input)

            # Fuse the features
            features = self.modality_specific_extractor.fuse_multimodal_features(text_features, vision_features)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Prepare modality info dictionary
        modality_info = {
            'modality': modality,
            'confidence_scores': confidence_scores,
            'complexity_score': complexity_score,
            'feature_shape': features.shape
        }

        return features, modality_info
    
    def get_optimal_processing_depth(self, complexity_score: float) -> int:
        """
        Determine optimal processing depth based on input complexity.
        
        Args:
            complexity_score: Complexity score between 0 and 1
            
        Returns:
            Optimal number of layers to process
        """
        # Use configuration parameters to determine depth range
        min_depth = int(self.config.num_hidden_layers * getattr(self.config, 'min_depth_ratio', 0.2))
        max_depth = self.config.num_hidden_layers
        
        # Calculate depth based on complexity
        depth_range = max_depth - min_depth
        optimal_depth = min_depth + int(complexity_score * depth_range)
        
        return max(min(optimal_depth, max_depth), min_depth)