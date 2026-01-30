"""
Tests for PaddleOCR-VL-1.5 Plugin

This module contains tests for the PaddleOCR-VL-1.5 plugin, verifying
loading, inference, and optimization integration.
"""

import unittest
import os
import torch
from PIL import Image
import numpy as np

# Adjust path to allow imports from src
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference_pio.models.paddleocr_vl_1_5.plugin import create_paddleocr_vl_1_5_plugin
from src.inference_pio.models.paddleocr_vl_1_5.config import PaddleOCRVL15Config
from src.inference_pio.models.paddleocr_vl_1_5.image_processing import OptimizedImageProcessor

class TestPaddleOCRVL15Plugin(unittest.TestCase):
    def setUp(self):
        self.plugin = create_paddleocr_vl_1_5_plugin()
        # Create a dummy image for testing
        self.dummy_image = Image.new('RGB', (100, 100), color='white')

    def test_initialization(self):
        """Test plugin initialization."""
        success = self.plugin.initialize()
        self.assertTrue(success)
        self.assertIsNotNone(self.plugin.config)
        self.assertIsNotNone(self.plugin.model_wrapper)

    def test_config_defaults(self):
        """Test configuration defaults."""
        self.plugin.initialize()
        config = self.plugin.config
        self.assertTrue(config.enable_paged_kv_cache)
        self.assertEqual(config.model_name, "PaddleOCR-VL-1.5")

    def test_image_preprocessing_logic(self):
        """Test image preprocessing specific logic (e.g. upscaling)."""
        self.plugin.initialize()
        # Instantiate processor directly to test logic
        processor = OptimizedImageProcessor(self.plugin.config)

        # Create a small image that should trigger upscaling for 'spotting'
        small_img = Image.new('RGB', (1000, 1000), color='white')

        # 1. Normal task - no resize
        processed_normal = processor.preprocess(small_img, task="ocr")
        self.assertEqual(processed_normal.size, (1000, 1000))

        # 2. Spotting task - should resize (upscale)
        processed_spotting = processor.preprocess(small_img, task="spotting")
        self.assertEqual(processed_spotting.size, (2000, 2000))

    @unittest.skipIf(not torch.cuda.is_available() and os.getenv("FORCE_REAL_MODEL") != "1", "Skipping real model load test unless FORCE_REAL_MODEL is set")
    def test_real_model_loading_and_inference(self):
        """
        Test loading the REAL model and running inference.
        WARNING: This requires downloading the model (few GBs).
        """
        print("Attempting to load real model (this may trigger download)...")
        self.plugin.initialize()

        # This will trigger download if not present
        model = self.plugin.load_model()
        self.assertIsNotNone(model)
        self.assertTrue(self.plugin.is_loaded)

        # Run inference
        inputs = {
            "image": self.dummy_image,
            "task": "ocr"
        }
        result = self.plugin.infer(inputs)
        print(f"Inference Result: {result}")
        self.assertIsInstance(result, str)

if __name__ == "__main__":
    unittest.main()
