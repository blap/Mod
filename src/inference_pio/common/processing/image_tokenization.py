"""
Generic Image Tokenization System for Vision-Language Models

This module implements a generic efficient image tokenization system for vision-language models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenericImageTokenizationConfig:
    """Generic configuration for image tokenization system."""

    # Basic image processing parameters
    image_size: int = 448  # Size of processed images
    patch_size: int = 14  # Size of image patches
    num_patches: int = 1024  # Calculated as (image_size // patch_size)^2

    # Normalization parameters (ImageNet defaults)
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

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

    # Model-specific parameters that can be overridden
    model_path: str = "placeholder_model"
    trust_remote_code: bool = True


class StandardImageProcessor:
    """
    Standard efficient image processor replacing transformers.AutoImageProcessor.
    Handles resizing, normalization, and tensor conversion efficiently.
    """
    def __init__(self, config: GenericImageTokenizationConfig):
        self.config = config
        self.mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(config.image_std).view(1, 3, 1, 1)

    def __call__(self, images: Union[Image.Image, List[Image.Image], torch.Tensor], return_tensors="pt", **kwargs):
        """
        Process images into model inputs.
        """
        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, torch.Tensor):
            # If tensor, assume it's [C, H, W] or [B, C, H, W] and unnormalized
            if images.dim() == 3:
                images = images.unsqueeze(0)
            return {"pixel_values": self._normalize(images)}

        # Process PIL images
        processed_tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize
            img = img.resize((self.config.image_size, self.config.image_size), resample=Image.BICUBIC)

            # To Tensor [C, H, W] and Scale to [0, 1]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            processed_tensors.append(img_tensor)

        # Stack [B, C, H, W]
        batch_tensor = torch.stack(processed_tensors)

        # Normalize
        normalized = self._normalize(batch_tensor)

        if return_tensors == "pt":
            return {"pixel_values": normalized}
        else:
            return {"pixel_values": normalized.numpy()}

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize batch tensor [B, C, H, W] using mean and std."""
        # Ensure mean/std are on same device as tensor
        if self.mean.device != tensor.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)

        return (tensor - self.mean) / self.std


class GenericImageTokenizer:
    """
    Generic efficient image tokenizer for vision-language models.
    Converts images to tokens with various optimization techniques.
    """

    def __init__(
        self,
        config: GenericImageTokenizationConfig,
        image_processor: Optional[StandardImageProcessor] = None,
    ):
        self.config = config

        # Initialize image processor if not provided
        if image_processor is None:
            self.image_processor = StandardImageProcessor(config)
        else:
            self.image_processor = image_processor

        # Calculate derived parameters
        self.num_patches_per_side = self.config.image_size // self.config.patch_size
        self.total_patches = self.num_patches_per_side**2

        # Initialize caches
        self.patch_cache = {} if self.config.enable_patch_caching else None
        self.compression_cache = {} if self.config.enable_compression else None

        # Performance metrics
        self.total_tokenization_time = 0.0
        self.num_tokenization_calls = 0

        logger.info(f"Initialized GenericImageTokenizer with config: {self.config}")

    def tokenize(
        self, image: Union[Image.Image, str, torch.Tensor], return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize an image into tokens for vision-language models.
        """
        start_time = time.time()

        # Load image if it's a path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Image must be PIL Image, path string, or tensor, got {type(image)}")

        # Process the image
        processed = self._process_image(image, return_tensors)

        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_tokenization_time += elapsed_time
        self.num_tokenization_calls += 1

        return processed

    def _process_image(
        self, image: Union[Image.Image, torch.Tensor], return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Internal method to process an image into tokens.
        """
        # Use the image processor to get pixel values
        processed = self.image_processor(images=image, return_tensors=return_tensors)

        # Apply optimizations based on config
        if self.config.enable_quantization:
            processed = self._quantize_image(processed)

        if self.config.enable_compression:
            processed = self._compress_image(processed)

        return processed

    def _quantize_image(
        self, processed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply quantization to the image for efficiency.
        """
        if self.config.quantization_bits == 8:
            pixel_values = processed["pixel_values"]
            # Clamp values to [-1, 1] range (if already normalized approx) and scale
            # Note: Standard normalization puts values roughly in [-2, 2].
            # We assume the user wants simple 8-bit quantization here.
            # Simplified for now: just float16 or similar might be better, but sticking to logic.
            # Using simple min/max scaling for 8bit mapping
            min_val, max_val = pixel_values.min(), pixel_values.max()
            scale = 255.0 / (max_val - min_val + 1e-6)
            pixel_values = ((pixel_values - min_val) * scale).byte()
            processed["pixel_values"] = pixel_values.float() # Dequant would require storing scale/min

        return processed

    def _compress_image(
        self, processed: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply compression to the image for storage efficiency.
        """
        pixel_values = processed["pixel_values"]

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        if self.config.compression_ratio < 1.0:
            _, _, height, width = pixel_values.shape
            target_size = int(height * (self.config.compression_ratio**0.5))
            if target_size < height and target_size < width:
                pixel_values = F.interpolate(
                    pixel_values,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )

        processed["pixel_values"] = pixel_values
        return processed

    def batch_tokenize(
        self,
        images: List[Union[Image.Image, str, torch.Tensor]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of images efficiently.
        """
        if not self.config.enable_batch_processing:
            batched_results = []
            for img in images:
                result = self.tokenize(img, return_tensors)
                batched_results.append(result["pixel_values"])

            if return_tensors == "pt":
                batched_pixel_values = torch.cat(batched_results, dim=0)
            else:
                batched_pixel_values = np.concatenate(batched_results, axis=0)

            return {"pixel_values": batched_pixel_values}

        start_time = time.time()

        # Use processor batch capability
        processed = self.image_processor(images, return_tensors=return_tensors)

        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_tokenization_time += elapsed_time
        self.num_tokenization_calls += 1

        return processed

    def tokenize_with_patches(
        self, image: Union[Image.Image, str, torch.Tensor], return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize an image and return patch-based tokens.
        """
        processed = self.tokenize(image, return_tensors)
        pixel_values = processed["pixel_values"]

        # Extract patches logic (simplified)
        if (
            pixel_values.shape[-1] == self.config.image_size
            and pixel_values.shape[-2] == self.config.image_size
        ):
            batch_size, channels, height, width = pixel_values.shape
            patch_h, patch_w = self.config.patch_size, self.config.patch_size
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w

            patches = pixel_values.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
            patches = patches.contiguous().view(
                batch_size, channels, num_patches_h * num_patches_w, patch_h, patch_w
            )
            patches = patches.transpose(1, 2)
            patch_vectors = patches.contiguous().view(
                batch_size, num_patches_h * num_patches_w, -1
            )
            processed["patch_tokens"] = patch_vectors
            processed["num_patches"] = num_patches_h * num_patches_w

        return processed

    def get_performance_stats(self) -> Dict[str, float]:
        if self.num_tokenization_calls > 0:
            avg_time = self.total_tokenization_time / self.num_tokenization_calls
        else:
            avg_time = 0.0

        return {
            "total_tokenization_time": self.total_tokenization_time,
            "num_tokenization_calls": self.num_tokenization_calls,
            "average_tokenization_time": avg_time,
        }

    def reset_performance_stats(self):
        self.total_tokenization_time = 0.0
        self.num_tokenization_calls = 0


class GenericEfficientImageProcessor:
    """
    Generic efficient image processor that can be used with any vision-language model.
    """

    def __init__(
        self,
        config: GenericImageTokenizationConfig,
        image_processor: Optional[StandardImageProcessor] = None,
    ):
        self.config = config
        self.tokenizer = GenericImageTokenizer(config, image_processor)
        self.enable_fast_resize = True
        self.enable_color_jittering = False

    def process(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        return_tensors: str = "pt",
        apply_augmentation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if apply_augmentation and self.enable_color_jittering:
            image = self._apply_augmentations(image)

        result = self.tokenizer.tokenize(image, return_tensors)
        return result

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        return image

    def batch_process(
        self,
        images: List[Union[Image.Image, str, torch.Tensor]],
        return_tensors: str = "pt",
        apply_augmentation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer.batch_tokenize(images, return_tensors)


def create_generic_image_tokenizer(
    model_path: Optional[str] = None,
    config: Optional[GenericImageTokenizationConfig] = None,
) -> GenericImageTokenizer:
    """
    Factory function to create a generic image tokenizer.
    Does NOT depend on external libraries anymore.
    """
    if config is None:
        config = GenericImageTokenizationConfig()
        if model_path:
            config.model_path = model_path

    logger.info(f"Creating generic image tokenizer with config: {config}")

    # Just create standard processor
    tokenizer = GenericImageTokenizer(config)
    return tokenizer


def apply_image_tokenization_to_model(
    model: nn.Module, tokenizer: GenericImageTokenizer
) -> nn.Module:
    model.image_tokenizer = tokenizer
    return model


def get_optimized_image_processor(
    model_path: Optional[str] = None,
) -> GenericEfficientImageProcessor:
    config = GenericImageTokenizationConfig()
    if model_path:
        config.model_path = model_path

    # No longer tries to load AutoImageProcessor
    tokenizer = create_generic_image_tokenizer(model_path, config)
    return GenericEfficientImageProcessor(config, tokenizer.image_processor)


__all__ = [
    "GenericImageTokenizationConfig",
    "GenericImageTokenizer",
    "GenericEfficientImageProcessor",
    "create_generic_image_tokenizer",
    "apply_image_tokenization_to_model",
    "get_optimized_image_processor",
]
