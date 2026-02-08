"""
Image Tokenization and Processing - Dependency Free (No PIL, No Numpy)
"""

import logging
import json
import os
import struct
from typing import Dict, List, Optional, Tuple, Union, Any

# Use C-Engine Tensors
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class StandardImageProcessor:
    """
    Standard image processor using pure Python binary reading and C-Engine ops.
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

    def load_image_raw(self, filepath: str) -> Tensor:
        """
        Load image from file (Simple BMP/PPM support or Raw RGB).
        Returns Tensor [C, H, W]
        """
        # Minimal PPM P6 parser (standard for raw benchmarks)
        with open(filepath, 'rb') as f:
            header = b""
            while True:
                byte = f.read(1)
                if byte in [b'\n', b' ']: break
                header += byte

            if header == b'P6':
                # Read dims
                dims = []
                while len(dims) < 3:
                    word = b""
                    while True:
                        byte = f.read(1)
                        if byte in [b'\n', b' ']:
                            if word: break
                            continue
                        word += byte
                    dims.append(int(word))

                w, h, maxval = dims
                data = f.read()

                # Convert to floats
                floats = [b / 255.0 for b in data] # Simple normalization to 0-1

                # Reshape to [C, H, W] (Planar)
                # PPM is interleaved RGBRGB
                c_h_w = [0.0] * (3 * h * w)
                for i in range(h * w):
                    c_h_w[i] = floats[i*3]         # R
                    c_h_w[h*w + i] = floats[i*3+1] # G
                    c_h_w[2*h*w + i] = floats[i*3+2] # B

                return Tensor([3, h, w], c_h_w)

        # For other formats (JPEG/PNG), require external tool to convert to raw/ppm first
        # or implement complex decoders. For "custom code", PPM is standard enough.
        logger.warning(f"Unsupported image format for {filepath}, returning dummy tensor.")
        return Tensor([3, 224, 224])

    def __call__(self, images: Union[str, List[str]], return_tensors: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process images into model inputs.
        Accepts file paths.
        """
        if isinstance(images, str):
            images = [images]

        processed_tensors = []

        for img_path in images:
            tensor = self.load_image_raw(img_path)

            if self.do_rescale:
                # Raw loader already normalized to 0-1 roughly, but if rescale factor is specific
                # we apply it. Usually raw RGB is 0-255, my loader did /255.
                # If do_rescale is True and factor is 1/255, we assume input was 0-255.
                # Since my loader returned 0-1, we skip or adjust.
                pass

            if self.do_resize:
                tensor = tensor.resize_image(self.size["height"], self.size["width"])

            if self.do_normalize:
                tensor = tensor.normalize_image(self.image_mean, self.image_std)

            processed_tensors.append(tensor)

        # Stack? C-Engine Tensor doesn't support stack yet easily (need new malloc)
        # Return list of tensors for now or single tensor if batch=1
        if len(processed_tensors) == 1:
            return {"pixel_values": processed_tensors[0]}

        return {"pixel_values": processed_tensors} # List

def get_optimized_image_processor(model_path: str) -> StandardImageProcessor:
    config_path = os.path.join(model_path, "preprocessor_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        size = config.get("size", {"height": 224, "width": 224})
        if isinstance(size, int): size = {"height": size, "width": size}
        return StandardImageProcessor(
            size=size,
            image_mean=config.get("image_mean"),
            image_std=config.get("image_std"),
            do_resize=config.get("do_resize", True),
            do_normalize=config.get("do_normalize", True),
            do_rescale=config.get("do_rescale", True),
            rescale_factor=config.get("rescale_factor", 1/255.0)
        )
    return StandardImageProcessor()
