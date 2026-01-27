"""
Efficient Image Tokenization System for Qwen3-VL-2B Model - Self-Contained Version

This module implements an efficient image tokenization system specifically optimized
for the Qwen3-VL-2B model. The system handles conversion of images to tokens with
various optimization techniques for vision tasks.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np
from dataclasses import dataclass
import hashlib
from io import BytesIO

logger = logging.getLogger(__name__)


@dataclass
class ImageTokenizationConfig:
    """Configuration for image tokenization system."""
    # Basic image processing parameters
    image_size: int = 448  # Size of processed images
    patch_size: int = 14   # Size of image patches
    num_patches: int = 1024  # Calculated as (image_size // patch_size)^2

    # Tokenization parameters
    max_image_tokens: int = 1024  # Maximum number of image tokens
    token_dim: int = 1024  # Dimension of each image token

    # Performance optimization parameters
    enable_patch_caching: bool = True  # Enable caching of processed patches
    enable_batch_processing: bool = True  # Enable batch processing
    enable_memory_efficient_processing: bool = True  # Enable memory-efficient processing

    # Quantization parameters for efficiency
    enable_quantization: bool = False  # Enable quantization for faster processing
    quantization_bits: int = 8  # Bits for quantization

    # Compression parameters
    enable_compression: bool = True  # Enable compression for storage efficiency
    compression_ratio: float = 0.5  # Compression ratio (0.0 to 1.0)


class ImageTokenizer:
    """
    Efficient image tokenizer for the Qwen3-VL-2B model.
    Converts images to tokens with various optimization techniques.
    """

    def __init__(self, config: ImageTokenizationConfig, image_processor: Optional[AutoImageProcessor] = None):
        self.config = config

        # Initialize image processor if not provided
        if image_processor is None:
            try:
                # Attempt to load a default image processor
                from transformers import AutoImageProcessor
                self.image_processor = AutoImageProcessor.from_pretrained(
                    "H:/Qwen3-VL-2B-Instruct",
                    trust_remote_code=True
                )
            except:
                # Fallback to a basic processor
                self.image_processor = self._create_basic_image_processor()
        else:
            self.image_processor = image_processor

        # Calculate derived parameters
        self.num_patches_per_side = self.config.image_size // self.config.patch_size
        self.total_patches = self.num_patches_per_side ** 2

        # Initialize caches
        self.patch_cache = {} if self.config.enable_patch_caching else None
        self.compression_cache = {} if self.config.enable_compression else None

        # Performance metrics
        self.total_tokenization_time = 0.0
        self.num_tokenization_calls = 0

        logger.info(f"Initialized ImageTokenizer with config: {self.config}")

    def _create_basic_image_processor(self):
        """Create a basic image processor if none is available."""
        self_obj = self  # Capture self reference

        class BasicImageProcessor:
            def __call__(self, images, return_tensors="pt", **kwargs):
                # Helper to get size from kwargs or config
                size = kwargs.get("size", {})
                height = size.get("height", self_obj.config.image_size)
                width = size.get("width", self_obj.config.image_size)

                # Convert PIL image to tensor
                if isinstance(images, Image.Image):
                    # Convert to RGB if needed
                    if images.mode != 'RGB':
                        images = images.convert('RGB')

                    # Resize image to expected size
                    if images.size != (width, height):
                        images = images.resize((width, height))

                    # Convert to tensor and normalize
                    image_array = np.array(images).astype(np.float32)
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW

                    # Normalize to [-1, 1] range
                    image_tensor = (image_tensor / 127.5) - 1.0

                    if return_tensors == "pt":
                        return {"pixel_values": image_tensor.unsqueeze(0)}  # Add batch dimension
                    else:
                        return {"pixel_values": image_tensor.numpy().unsqueeze(0)}
                elif isinstance(images, torch.Tensor):
                    # If it's already a tensor, ensure proper shape
                    if images.dim() == 3:  # (C, H, W)
                        images = images.unsqueeze(0)  # Add batch dimension
                    elif images.dim() != 4:  # Not (B, C, H, W)
                        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {images.dim()}")

                    return {"pixel_values": images}

        return BasicImageProcessor()

    def tokenize(self, image: Union[Image.Image, str, torch.Tensor],
                 return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Tokenize an image into tokens for the Qwen3-VL-2B model.

        Args:
            image: Input image (PIL Image, path string, or tensor)
            return_tensors: Format for returned tensors ("pt", "np", etc.)

        Returns:
            Dictionary containing tokenized image data
        """
        start_time = time.time()

        # Load image if it's a path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, torch.Tensor):
            # If it's already a tensor, ensure it's in the right format
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Image must be PIL Image, path string, or tensor, got {type(image)}")

        # Process the image
        processed = self._process_image_optimized(image, return_tensors)

        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_tokenization_time += elapsed_time
        self.num_tokenization_calls += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Image tokenization took {elapsed_time:.4f}s for image size {image.size if hasattr(image, 'size') else image.shape}")

        return processed

    def _process_image(self, image: Union[Image.Image, torch.Tensor],
                      return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Legacy method kept for compatibility. Redirects to optimized version.
        """
        return self._process_image_optimized(image, return_tensors)

    def _process_image_optimized(self, image: Union[Image.Image, torch.Tensor],
                      return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Internal method to process an image into tokens with optimizations.

        Args:
            image: Input image (PIL Image or tensor)
            return_tensors: Format for returned tensors

        Returns:
            Dictionary containing processed image data
        """
        # Prepare processing arguments
        process_kwargs = {}

        # Optimization 1: Calculate target size beforehand if compression is enabled
        # This avoids resizing to large size then downscaling
        if self.config.enable_compression and self.config.compression_ratio < 1.0:
            target_size = int(self.config.image_size * (self.config.compression_ratio ** 0.5))
            process_kwargs["size"] = {"height": target_size, "width": target_size}
        else:
             process_kwargs["size"] = {"height": self.config.image_size, "width": self.config.image_size}

        # Use the image processor to get pixel values
        processed = self.image_processor(images=image, return_tensors=return_tensors, **process_kwargs)

        # Apply optimizations based on config
        if self.config.enable_quantization:
            processed = self._quantize_image(processed)

        # Compression is now largely handled during resizing (Optimization 1),
        # but _compress_image might still be needed if not using BasicImageProcessor
        # or for non-spatial compression
        if self.config.enable_compression:
            processed = self._compress_image(processed)

        # The Qwen models return pixel_values in a special format (patch-based)
        # The processed result may include additional keys like 'image_grid_thw'
        # We need to ensure the main pixel_values tensor is properly handled
        pixel_values = processed['pixel_values']

        # For Qwen models, the pixel_values might already be in the correct format
        # (i.e., patch-based format ready for the vision encoder)
        # Just return the processed result as is, but ensure it's in the right format
        if pixel_values.dim() == 1:  # If it's somehow a 1D tensor, return as is
            return processed
        elif pixel_values.dim() == 2:  # Expected for Qwen models: (num_patches, patch_dim)
            # This is the expected format for Qwen vision models
            # Each row represents a patch of the image
            return processed
        elif pixel_values.dim() == 3:  # (batch, num_patches, patch_dim)
            # Batched format for multiple images
            return processed
        elif pixel_values.dim() == 4:  # (batch, channels, height, width) - traditional format
            # Traditional image format, may need to be converted to patches
            # For now, just return as is
            return processed
        else:
            # Unexpected dimensions, return as is
            return processed

    def _quantize_image(self, processed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply quantization to the image for efficiency using in-place operations (Optimization 3).

        Args:
            processed: Processed image data

        Returns:
            Quantized image data
        """
        pixel_values = processed['pixel_values']

        if self.config.quantization_bits == 8:
            # Quantize to 8-bit
            # Clamp values to [-1, 1] range and scale to [0, 255]
            pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
            # Use in-place operations to reduce memory overhead
            pixel_values.add_(1.0).mul_(127.5).round_().byte()
            # Convert back to float in [0, 255] range (this creates new tensor, unavoidable for float)
            processed['pixel_values'] = pixel_values.float()

        elif self.config.quantization_bits == 4:
            # Quantize to 4-bit (simplified)
            pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
            pixel_values.add_(1.0).mul_(7.5).round_().char()  # 0-15 range
            processed['pixel_values'] = pixel_values.float()

        return processed

    def _compress_image(self, processed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply compression to the image for storage efficiency.

        Args:
            processed: Processed image data

        Returns:
            Compressed image data
        """
        # For now, implement a simple spatial compression
        pixel_values = processed['pixel_values']

        # Ensure pixel_values has the correct shape (batch, channels, height, width)
        if pixel_values.dim() == 3:
            # Add batch dimension if missing
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.dim() != 4:
            # If it's not 3D or 4D, return as is
            processed['pixel_values'] = pixel_values
            return processed

        if self.config.compression_ratio < 1.0:
            # Apply spatial downsampling based on compression ratio
            # Optimization: Only interpolate if dimensions don't match expectation
            # (Allows pre-resized images from _process_image_optimized to skip this)
            _, _, height, width = pixel_values.shape
            expected_size = int(self.config.image_size * (self.config.compression_ratio ** 0.5))

            # Tolerance of 1 pixel due to rounding
            if abs(height - expected_size) > 1 or abs(width - expected_size) > 1:
                 # Check if we are already smaller (don't upscale)
                 if expected_size < height and expected_size < width:
                    pixel_values = F.interpolate(
                        pixel_values,
                        size=(expected_size, expected_size),
                        mode='bilinear',
                        align_corners=False
                    )

        processed['pixel_values'] = pixel_values
        return processed

    def batch_tokenize(self, images: List[Union[Image.Image, str, torch.Tensor]],
                      return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of images efficiently (Optimization 2).

        Args:
            images: List of input images
            return_tensors: Format for returned tensors

        Returns:
            Dictionary containing batched tokenized image data
        """
        if not self.config.enable_batch_processing:
            # Fallback to sequential processing
            batched_results = []
            for img in images:
                result = self.tokenize(img, return_tensors)
                batched_results.append(result['pixel_values'])

            if return_tensors == "pt":
                batched_pixel_values = torch.cat(batched_results, dim=0)
            else:
                batched_pixel_values = np.concatenate(batched_results, axis=0)

            return {'pixel_values': batched_pixel_values}

        start_time = time.time()

        # Determine target size
        if self.config.enable_compression and self.config.compression_ratio < 1.0:
            target_size = int(self.config.image_size * (self.config.compression_ratio ** 0.5))
        else:
            target_size = self.config.image_size

        # Optimization 2: Vectorized processing for PIL images
        pil_images = [img for img in images if isinstance(img, Image.Image)]
        if len(pil_images) == len(images) and len(images) > 0:
            # All are PIL images, we can optimize
            resized_images = []
            for img in pil_images:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if img.size != (target_size, target_size):
                    resized_images.append(img.resize((target_size, target_size)))
                else:
                    resized_images.append(img)

            # Convert to numpy stack first (faster than list of tensors)
            try:
                # This requires images to be same size, which we ensured above
                # Optimization: Direct numpy stack is fast
                image_stack = np.stack([np.array(img) for img in resized_images])

                # To Tensor and Permute
                batch_tensor = torch.from_numpy(image_stack).permute(0, 3, 1, 2).float()

                # Normalize (Vectorized)
                # In-place operations
                batch_tensor.div_(127.5).sub_(1.0)

                batched_pixel_values = batch_tensor
            except Exception as e:
                logger.warning(f"Vectorized batch processing failed: {e}. Falling back to loop.")
                # Fallback to loop if numpy stack fails
                batched_pixel_values = self._batch_tokenize_loop(images, return_tensors)
        else:
            # Mixed types or no images, fall back to loop
            batched_pixel_values = self._batch_tokenize_loop(images, return_tensors)

        # Apply post-processing optimizations (quantization/compression)
        # Note: Compression (resizing) was already handled above for PIL path
        # Quantization needs to be applied
        if self.config.enable_quantization:
             batched_pixel_values = self._quantize_batch(batched_pixel_values)

        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_tokenization_time += elapsed_time
        self.num_tokenization_calls += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Batch image tokenization took {elapsed_time:.4f}s for {len(images)} images")

        return {'pixel_values': batched_pixel_values}

    def _batch_tokenize_loop(self, images, return_tensors):
        """Fallback loop implementation for batch tokenization."""
        processed_list = []
        for img in images:
            processed = self.tokenize(img, return_tensors)
            processed_list.append(processed['pixel_values'])

        if return_tensors == "pt":
            return torch.cat(processed_list, dim=0) if processed_list[0].dim() == 4 else torch.stack(processed_list, dim=0)
        else:
            return np.concatenate(processed_list, axis=0) if processed_list[0].ndim == 4 else np.stack(processed_list, axis=0)

    def _quantize_batch(self, pixel_values):
        """Helper to quantize a batch tensor."""
        if self.config.quantization_bits == 8:
            pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
            pixel_values.add_(1.0).mul_(127.5).round_().byte()
            return pixel_values.float()
        elif self.config.quantization_bits == 4:
            pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
            pixel_values.add_(1.0).mul_(7.5).round_().char()
            return pixel_values.float()
        return pixel_values

    def tokenize_with_patches(self, image: Union[Image.Image, str, torch.Tensor],
                             return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Tokenize an image and return patch-based tokens.

        Args:
            image: Input image
            return_tensors: Format for returned tensors

        Returns:
            Dictionary containing patch-based tokenized data
        """
        # First, process the image normally
        processed = self.tokenize(image, return_tensors)
        pixel_values = processed['pixel_values']

        # Extract patches if needed
        if pixel_values.shape[-1] == self.config.image_size and pixel_values.shape[-2] == self.config.image_size:
            # Extract patches using unfold operation
            batch_size, channels, height, width = pixel_values.shape

            # Calculate patch size
            patch_h, patch_w = self.config.patch_size, self.config.patch_size
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w

            # Unfold the image to extract patches
            patches = pixel_values.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
            patches = patches.contiguous().view(batch_size, channels, num_patches_h * num_patches_w, patch_h, patch_w)
            patches = patches.transpose(1, 2)  # Shape: (batch, num_patches, channels, patch_h, patch_w)

            # Flatten patches to vectors
            patch_vectors = patches.contiguous().view(batch_size, num_patches_h * num_patches_w, -1)

            # Apply linear projection to match token dimension
            if hasattr(self, 'patch_projection'):
                patch_tokens = self.patch_projection(patch_vectors)
            else:
                # Create a simple linear projection if not available
                patch_tokens = patch_vectors  # For now, just return the patch vectors

            processed['patch_tokens'] = patch_tokens
            processed['num_patches'] = num_patches_h * num_patches_w

        return processed

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the tokenization system.

        Returns:
            Dictionary containing performance metrics
        """
        if self.num_tokenization_calls > 0:
            avg_time = self.total_tokenization_time / self.num_tokenization_calls
        else:
            avg_time = 0.0

        return {
            'total_tokenization_time': self.total_tokenization_time,
            'num_tokenization_calls': self.num_tokenization_calls,
            'average_tokenization_time': avg_time
        }

    def reset_performance_stats(self):
        """
        Reset performance statistics.
        """
        self.total_tokenization_time = 0.0
        self.num_tokenization_calls = 0


class EfficientImageProcessor:
    """
    Efficient image processor that integrates with the Qwen3-VL-2B model.
    Combines image preprocessing and tokenization for optimal performance.
    """

    def __init__(self, config: ImageTokenizationConfig,
                 image_processor: Optional[AutoImageProcessor] = None):
        self.config = config
        self.tokenizer = ImageTokenizer(config, image_processor)

        # Additional optimizations
        self.enable_fast_resize = True
        self.enable_color_jittering = False  # Only for training

    def process(self, image: Union[Image.Image, str, torch.Tensor],
               return_tensors: str = "pt",
               apply_augmentation: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process an image with all optimizations applied.

        Args:
            image: Input image
            return_tensors: Format for returned tensors
            apply_augmentation: Whether to apply augmentation (for training)

        Returns:
            Dictionary containing processed image data
        """
        # Apply preprocessing
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Apply augmentations if needed
        if apply_augmentation and self.enable_color_jittering:
            image = self._apply_augmentations(image)

        # Tokenize the image
        result = self.tokenizer.tokenize(image, return_tensors)

        return result

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentations to the image (for training).

        Args:
            image: Input image

        Returns:
            Augmented image
        """
        # For now, just return the original image
        # In a real implementation, you'd apply random crops, flips, etc.
        return image

    def batch_process(self, images: List[Union[Image.Image, str, torch.Tensor]],
                     return_tensors: str = "pt",
                     apply_augmentation: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process a batch of images efficiently.

        Args:
            images: List of input images
            return_tensors: Format for returned tensors
            apply_augmentation: Whether to apply augmentation

        Returns:
            Dictionary containing batched processed image data
        """
        # Apply preprocessing to all images
        processed_images = []
        for img in images:
            processed = self.process(img, return_tensors, apply_augmentation)
            processed_images.append(processed['pixel_values'])

        # Stack the results
        if return_tensors == "pt":
            batched_pixel_values = torch.stack(processed_images, dim=0)
        else:
            batched_pixel_values = np.stack(processed_images, axis=0)

        return {'pixel_values': batched_pixel_values}


def create_image_tokenizer(model_path: Optional[str] = None,
                          config: Optional[ImageTokenizationConfig] = None) -> ImageTokenizer:
    """
    Factory function to create an image tokenizer for Qwen3-VL-2B.

    Args:
        model_path: Path to the Qwen3-VL-2B model (optional)
        config: Image tokenization configuration (optional)

    Returns:
        ImageTokenizer instance
    """
    if config is None:
        config = ImageTokenizationConfig()
    elif isinstance(config, dict):
        # Convert dict to ImageTokenizationConfig
        config_obj = ImageTokenizationConfig()
        for k, v in config.items():
            if hasattr(config_obj, k):
                setattr(config_obj, k, v)
        config = config_obj

    logger.info(f"Creating image tokenizer with config: {config}")

    # Load image processor if model path is provided
    image_processor = None
    if model_path:
        try:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Could not load image processor from {model_path}: {e}")

    # Create and return the tokenizer
    tokenizer = ImageTokenizer(config, image_processor)

    logger.info("Image tokenizer created successfully")
    return tokenizer


def apply_image_tokenization_to_model(model: nn.Module,
                                    tokenizer: ImageTokenizer) -> nn.Module:
    """
    Apply image tokenization optimizations to the model.

    Args:
        model: The model to optimize
        tokenizer: The image tokenizer to attach

    Returns:
        Optimized model with image tokenization capabilities
    """
    logger.info("Applying image tokenization optimizations to model...")

    # Attach the tokenizer to the model
    model.image_tokenizer = tokenizer

    # Optionally, we could add preprocessing hooks here
    # For now, we just attach the tokenizer as an attribute

    logger.info("Image tokenization optimizations applied successfully")
    return model


def get_optimized_image_processor(model_path: Optional[str] = None) -> EfficientImageProcessor:
    """
    Get an optimized image processor for the Qwen3-VL-2B model.

    Args:
        model_path: Path to the Qwen3-VL-2B model (optional)

    Returns:
        EfficientImageProcessor instance
    """
    config = ImageTokenizationConfig()
    tokenizer = create_image_tokenizer(model_path, config)
    return EfficientImageProcessor(config, tokenizer.image_processor)


__all__ = [
    "ImageTokenizationConfig",
    "ImageTokenizer",
    "EfficientImageProcessor",
    "create_image_tokenizer",
    "apply_image_tokenization_to_model",
    "get_optimized_image_processor"
]