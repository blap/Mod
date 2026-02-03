#!/usr/bin/env python3
"""
Inference CLI Tool

This script allows running inference on various models using the Inference-PIO system.
"""

import argparse
import logging
import os
import sys
import time

from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from inference_pio.model_factory import ModelFactory


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    setup_logging()
    logger = logging.getLogger("InferenceCLI")

    parser = argparse.ArgumentParser(
        description="Run inference with Inference-PIO models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model name (available: {', '.join(ModelFactory.list_supported_models())})",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )
    parser.add_argument(
        "--image", type=str, help="Path to image file (for multimodal models)"
    )
    parser.add_argument(
        "--virtual-execution",
        action="store_true",
        help="Enable virtual execution (simulation) mode",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to run on (cpu, cuda, auto)"
    )

    args = parser.parse_args()

    # Create model
    logger.info(f"Initializing model: {args.model}")
    model_config = {
        "device": args.device,
        "enable_virtual_execution": args.virtual_execution,
    }

    plugin = ModelFactory.create_model(args.model, config=model_config)

    if not plugin:
        logger.error(f"Failed to load model: {args.model}")
        sys.exit(1)

    try:
        # Prepare input
        input_data = args.prompt

        if args.image:
            if "vl" not in args.model.lower():
                logger.warning(
                    f"Image provided but model {args.model} might not be multimodal."
                )

            try:
                image = Image.open(args.image)
                input_data = {"text": args.prompt, "image": image}
                logger.info(f"Loaded image from {args.image}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                sys.exit(1)

        # Run inference
        logger.info("Running inference...")
        start_time = time.time()

        # Virtual execution handling
        if args.virtual_execution:
            if hasattr(plugin, "execute_with_distributed_simulation"):
                # Some plugins use this name
                result = plugin.execute_with_distributed_simulation(input_data)
            elif hasattr(plugin, "execute_with_virtual_execution"):
                result = plugin.execute_with_virtual_execution(input_data)
            else:
                logger.warning(
                    f"Plugin {args.model} does not support virtual execution explicitly, using infer()"
                )
                result = plugin.infer(input_data)
        else:
            if isinstance(input_data, dict) and "vl" in args.model.lower():
                # Qwen3-VL specific infer signature handles dict
                result = plugin.infer(input_data)
            elif isinstance(input_data, str):
                # Text models
                if hasattr(plugin, "generate_text"):
                    result = plugin.generate_text(
                        input_data, max_new_tokens=args.max_tokens
                    )
                else:
                    result = plugin.infer(input_data)
            else:
                result = plugin.infer(input_data)

        end_time = time.time()

        print("\n" + "=" * 50)
        print(f"OUTPUT ({end_time - start_time:.2f}s):")
        print("=" * 50)
        print(result)
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if hasattr(plugin, "cleanup"):
            plugin.cleanup()


if __name__ == "__main__":
    main()
