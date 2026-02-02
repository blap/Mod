"""
Main entry point for the Inference-PIO package.

This module provides the command-line interface for the Inference-PIO system
with self-contained plugins. Each model plugin is completely independent
with its own configuration, tests, and benchmarks.
"""

import argparse
import sys
from typing import List, Optional

from . import (
    create_glm_4_7_flash_plugin,
    create_qwen3_0_6b_plugin,
    create_qwen3_4b_instruct_2507_plugin,
    create_qwen3_coder_30b_plugin,
    create_qwen3_vl_2b_instruct_plugin,
)
from .plugins.manager import get_plugin_manager


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Inference-PIO CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="inference-pio",
        description="Inference-PIO: Self-Contained Plugin Architecture for Advanced Model Inference",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List plugins command
    list_parser = subparsers.add_parser("list", help="List available plugins")

    # Run plugin command
    run_parser = subparsers.add_parser("run", help="Run a plugin")
    run_parser.add_argument(
        "plugin_name",
        choices=[
            "glm_4_7_flash",
            "qwen3_coder_30b",
            "qwen3_vl_2b",
            "qwen3_4b_instruct_2507",
            "qwen3_0_6b",
        ],
        help="Name of the plugin to run",
    )
    run_parser.add_argument("--input", "-i", required=True, help="Input for the plugin")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "list":
        pm = get_plugin_manager()
        plugins = pm.list_plugins()
        print("Available plugins:")
        for plugin in plugins:
            print(f"  - {plugin}")
        return 0

    elif parsed_args.command == "run":
        # Create and run the specified plugin
        if parsed_args.plugin_name == "glm_4_7_flash":
            plugin = create_glm_4_7_flash_plugin()
        elif parsed_args.plugin_name == "qwen3_coder_30b":
            plugin = create_qwen3_coder_30b_plugin()
        elif parsed_args.plugin_name == "qwen3_vl_2b":
            plugin = create_qwen3_vl_2b_instruct_plugin()
        elif parsed_args.plugin_name == "qwen3_4b_instruct_2507":
            plugin = create_qwen3_4b_instruct_2507_plugin()
        elif parsed_args.plugin_name == "qwen3_0_6b":
            plugin = create_qwen3_0_6b_plugin()
        else:
            print(f"Unknown plugin: {parsed_args.plugin_name}")
            return 1

        try:
            plugin.initialize()
            result = plugin.infer(parsed_args.input)
            print(result)
            plugin.cleanup()
            return 0
        except Exception as e:
            print(f"Error running plugin: {e}", file=sys.stderr)
            return 1

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
