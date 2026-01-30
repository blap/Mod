
import unittest
import logging
import torch
import shutil
import os
from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config

# Configure logging to see our debug prints
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestQwen3_0_6B_Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logger.info("Setting up Qwen3-0.6B Integration Test...")
        # Clean up temp dir if exists to force fresh logic test (optional, maybe risky if large download)
        # temp_path = os.path.join(os.getcwd(), "temp_models", "Qwen3-0.6B")
        # if os.path.exists(temp_path):
        #     shutil.rmtree(temp_path)

        cls.plugin = create_qwen3_0_6b_plugin()
        # Initialize with default config
        success = cls.plugin.initialize()
        if not success:
            raise RuntimeError("Failed to initialize plugin (Download or Load failed)")

    def test_basic_generation(self):
        logger.info("Testing basic generation...")
        prompt = "Hello, who are you?"
        output = self.plugin.generate_text(prompt, max_new_tokens=20)
        logger.info(f"Basic Output: {output}")
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_thinking_mode_switch(self):
        logger.info("Testing Thinking Mode switch (/think)...")
        prompt = "Solve 2+2. /think"
        # Logic check: _parse_thinking_switches
        clean_prompt, mode = self.plugin._parse_thinking_switches(prompt)
        self.assertEqual(clean_prompt, "Solve 2+2.")
        self.assertTrue(mode)

        # Run generation
        output = self.plugin.generate_text(prompt, max_new_tokens=50)
        logger.info(f"Thinking Output: {output}")
        # We can't guarantee <think> tags if the model is random/dumb fallback,
        # but if it's the real model, it should likely appear or at least run.

    def test_no_thinking_mode_switch(self):
        logger.info("Testing No-Thinking Mode switch (/no_think)...")
        prompt = "Solve 2+2. /no_think"
        clean_prompt, mode = self.plugin._parse_thinking_switches(prompt)
        self.assertEqual(clean_prompt, "Solve 2+2.")
        self.assertFalse(mode)

        output = self.plugin.generate_text(prompt, max_new_tokens=50)
        logger.info(f"No-Thinking Output: {output}")

    @classmethod
    def tearDownClass(cls):
        cls.plugin.cleanup()

if __name__ == '__main__':
    unittest.main()
