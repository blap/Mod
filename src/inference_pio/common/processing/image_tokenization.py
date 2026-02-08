"""
Image Tokenization and Processing
Standard efficient image processor replacing transformers.AutoImageProcessor.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from PIL import Image

# Try importing torch, but don't fail if missing
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class StandardImageProcessor:
    """
    Standard image processor that uses PIL and Numpy/Torch for efficient processing.
    """

    def __init__(self, size: Dict[str, int] = None, image_mean: List[float] = None,
                 image_std: List[float] = None, do_resize: bool = True,
                 do_normalize: bool = True, do_rescale: bool = True,
                 rescale_factor: float = 1/255.0):
        self.size = size or {"height": 224, "width": 224}
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor

    def __call__(self, images: Union[Image.Image, List[Image.Image]], return_tensors: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process images into model inputs.
        """
        if isinstance(images, Image.Image):
            images = [images]

        processed_images = []

        for img in images:
            # 1. Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 2. Resize
            if self.do_resize:
                img = img.resize((self.size["width"], self.size["height"]), resample=Image.BICUBIC)

            # 3. To Numpy
            img_array = np.array(img).astype(np.float32)

            # 4. Rescale
            if self.do_rescale:
                img_array = img_array * self.rescale_factor

            # 5. Normalize
            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)
                std = np.array(self.image_std, dtype=np.float32)
                img_array = (img_array - mean) / std

            # 6. Channel First [H, W, C] -> [C, H, W]
            img_array = img_array.transpose(2, 0, 1)

            processed_images.append(img_array)

        # Stack
        batch = np.stack(processed_images, axis=0)

        if return_tensors == "pt":
            if torch:
                return {"pixel_values": torch.from_numpy(batch)}
            else:
                logger.warning("Torch not available, returning numpy array instead of pt tensors")
                return {"pixel_values": batch}
        elif return_tensors == "np":
            return {"pixel_values": batch}

        # Default to whatever matches the environment or caller expectation
        # If Torch is requested but missing, we return numpy.
        return {"pixel_values": batch}

def get_optimized_image_processor(model_path: str) -> StandardImageProcessor:
    """
    Factory to create image processor from config.
    """
    config_path = os.path.join(model_path, "preprocessor_config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract params
        size = config.get("size", {"height": 224, "width": 224})
        if isinstance(size, int):
            size = {"height": size, "width": size}

        return StandardImageProcessor(
            size=size,
            image_mean=config.get("image_mean"),
            image_std=config.get("image_std"),
            do_resize=config.get("do_resize", True),
            do_normalize=config.get("do_normalize", True),
            do_rescale=config.get("do_rescale", True),
            rescale_factor=config.get("rescale_factor", 1/255.0)
        )
    else:
        logger.warning(f"Preprocessor config not found at {config_path}, using defaults.")
        return StandardImageProcessor()
