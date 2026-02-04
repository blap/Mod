"""
Inference-PIO Main Entry Point

This module provides the main entry point for the Inference-PIO system.
"""

import argparse
import logging
import os
import sys
from typing import Any

# Add the src directory to the path to allow imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inference_pio.core.model_factory import ModelFactory, create_model

__version__ = "1.0.0"
__author__ = "Inference-PIO Team"

logger = logging.getLogger(__name__)

def setup_logging(debug: bool):
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def handle_list(args: argparse.Namespace):
    """Handle list command."""
    models = ModelFactory.list_supported_models()
    print("Available Models:")
    for model in models:
        print(f"  - {model}")

def handle_run(args: argparse.Namespace):
    """Handle run command."""
    print(f"Loading model: {args.model}")
    try:
        model = create_model(args.model)

        print("Initializing model...")
        # We initialize with default configuration.
        # In a real scenario, we might want to pass more config options here.
        model.initialize()

        print(f"Running inference with prompt: '{args.prompt}'")
        print("-" * 40)

        if hasattr(model, 'generate_text'):
            result = model.generate_text(args.prompt)
        elif hasattr(model, 'infer'):
            result = model.infer(args.prompt)
        else:
            print("Error: Model does not support 'generate_text' or 'infer' methods.")
            return

        print(result)
        print("-" * 40)

    except Exception as e:
        logger.error(f"Failed to run model: {e}")
        if args.debug:
            raise
        else:
            print(f"Error running model: {e}")

def handle_chat(args: argparse.Namespace):
    """Handle chat command."""
    print(f"Loading model: {args.model}")
    try:
        model = create_model(args.model)
        model.initialize()

        print("\nStarting interactive chat session.")
        print("Type 'exit' or 'quit' to end the session.")
        print("-" * 40)

        # Check if model supports chat_completion, otherwise fallback to generate_text
        use_chat = hasattr(model, 'chat_completion')
        messages = []

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ('exit', 'quit'):
                    break

                if use_chat:
                    messages.append({"role": "user", "content": user_input})
                    response = model.chat_completion(messages)
                    messages.append({"role": "assistant", "content": response})
                elif hasattr(model, 'generate_text'):
                     # Simple one-shot for models that don't support chat history management
                     response = model.generate_text(user_input)
                else:
                    response = "Model does not support text generation."

                print(f"AI: {response}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nChat session ended.")

    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        if args.debug:
            raise
        else:
            print(f"Error starting chat: {e}")

def handle_info(args: argparse.Namespace):
    """Handle info command."""
    try:
        model = create_model(args.model)

        print(f"\nModel Information for {args.model}:")
        print("-" * 40)

        # Print metadata if available
        if hasattr(model, 'metadata'):
            print(f"Name: {model.metadata.name}")
            print(f"Version: {model.metadata.version}")
            print(f"Author: {model.metadata.author}")
            print(f"Type: {model.metadata.plugin_type.value}")
            print(f"Description: {model.metadata.description}")

        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print("\nDetails:")
            for key, value in info.items():
                if key not in ['name', 'description']: # Avoid duplication
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        # Print docstring if metadata didn't cover it or for extra detail
        if model.__doc__ and not getattr(model, 'metadata', None):
             print("\nDocstring:")
             print(model.__doc__.strip())

    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        if args.debug:
            raise
        else:
            print(f"Error getting info: {e}")

def handle_config(args: argparse.Namespace):
    """Handle config command."""
    # src_dir is already defined at module level but we need to be careful
    # src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # We can rely on relative path from this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'configs')

    if args.action == 'list':
        print(f"Configuration files in {config_dir}:")
        if os.path.exists(config_dir):
            files = [f for f in os.listdir(config_dir) if f.endswith(('.json', '.yaml', '.yml', '.ini', '.py')) and not f.startswith('__')]
            for f in files:
                print(f"  - {f}")
        else:
            print("Config directory not found.")

    elif args.action == 'view':
        if not args.file:
             print("Error: --file argument is required for view action.")
             return

        filepath = os.path.join(config_dir, args.file)
        # basic path traversal protection
        if '..' in args.file or args.file.startswith('/'):
             print("Error: Invalid file name.")
             return

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print(f"Content of {args.file}:")
            print("-" * 40)
            try:
                with open(filepath, 'r') as f:
                    print(f.read())
            except Exception as e:
                print(f"Error reading file: {e}")
            print("-" * 40)
        else:
            print(f"Error: File {args.file} not found in config directory.")

def handle_benchmark(args: argparse.Namespace):
    """Handle benchmark command."""
    print(f"Running benchmark suite: {args.suite}")
    try:
        from inference_pio.benchmarks.scripts.standardized_runner import run_standardized_benchmarks
        run_standardized_benchmarks(benchmark_suite=args.suite)
    except ImportError as e:
        logger.error(f"Failed to import benchmark runner: {e}")
        print(f"Error: Could not import benchmark runner. {e}")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.debug:
            raise
        else:
            print(f"Error running benchmark: {e}")

def main():
    """Main entry point for the Inference-PIO CLI."""
    parser = argparse.ArgumentParser(
        description="Inference-PIO: Self-Contained Plugin Architecture for Advanced Model Inference"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="version", version=f"Inference-PIO {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: list
    parser_list = subparsers.add_parser("list", help="List available models")
    parser_list.set_defaults(func=handle_list)

    # Command: run
    parser_run = subparsers.add_parser("run", help="Run single inference")
    parser_run.add_argument("--model", required=True, help="Model name (e.g., qwen3-0.6b)")
    parser_run.add_argument("--prompt", required=True, help="Input prompt")
    parser_run.set_defaults(func=handle_run)

    # Command: chat
    parser_chat = subparsers.add_parser("chat", help="Start interactive chat session")
    parser_chat.add_argument("--model", required=True, help="Model name")
    parser_chat.set_defaults(func=handle_chat)

    # Command: info
    parser_info = subparsers.add_parser("info", help="Show model information")
    parser_info.add_argument("--model", required=True, help="Model name")
    parser_info.set_defaults(func=handle_info)

    # Command: config
    parser_config = subparsers.add_parser("config", help="Manage configuration")
    parser_config.add_argument("action", choices=["list", "view"], help="Action to perform")
    parser_config.add_argument("--file", help="Config file name (required for view)")
    parser_config.set_defaults(func=handle_config)

    # Command: benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run benchmarks")
    parser_bench.add_argument("--suite", default="full", choices=["full", "performance", "accuracy"], help="Benchmark suite to run")
    parser_bench.set_defaults(func=handle_benchmark)

    args = parser.parse_args()

    setup_logging(args.debug)

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'config' and args.action == 'view' and not args.file:
             parser.error("the following arguments are required: --file")

        args.func(args)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        if args.debug:
            logger.exception("An error occurred:")
        else:
            print(f"Error: {e}")
            print("Use --debug for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
