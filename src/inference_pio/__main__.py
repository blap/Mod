import argparse
import logging
import os
import sys
from datetime import datetime

# Add the src directory to the path to allow imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inference_pio.core.model_factory import ModelFactory, create_model
from inference_pio.core.tools.system_check import (
    perform_system_check,
    print_system_check_report
)
from inference_pio.core.tools.rich_utils import setup_rich_logging, console
from inference_pio.core.tools.init_wizard import interactive_init
from inference_pio.core.tools.cleaner import clean_project
from rich.table import Table
from rich.panel import Panel

__version__ = "1.0.0"
__author__ = "Inference-PIO Team"

logger = logging.getLogger(__name__)


def setup_logging(debug: bool):
    """Configure logging based on debug flag."""
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs'
    )
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"inference_pio_{timestamp}.log")
    setup_rich_logging(debug=debug, log_file=log_file)


def handle_list(args: argparse.Namespace):
    """Handle list command."""
    models = ModelFactory.list_supported_models()
    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")

    for model in models:
        table.add_row(model, "Supported")

    console.print(table)


def handle_run(args: argparse.Namespace):
    """Handle run command."""
    console.print(f"[bold blue]Loading model:[/bold blue] {args.model}")
    try:
        model = create_model(args.model)

        console.print("[yellow]Initializing model...[/yellow]")
        with console.status("[bold green]Loading weights...[/bold green]") as st:
            # We initialize with default configuration.
            model.initialize()
            st.update("[bold green]Model loaded![/bold green]")

        console.print(
            Panel(
                args.prompt,
                title="[bold]Input Prompt[/bold]",
                border_style="blue"
            )
        )

        with console.status("[bold cyan]Generating...[/bold cyan]", spinner="dots"):
            if hasattr(model, 'generate_text'):
                result = model.generate_text(args.prompt)
            elif hasattr(model, 'infer'):
                result = model.infer(args.prompt)
            else:
                console.print(
                    "[bold red]Error: Model does not support "
                    "'generate_text' or 'infer' methods.[/bold red]"
                )
                return

        console.print(
            Panel(
                str(result),
                title="[bold]Model Output[/bold]",
                border_style="green"
            )
        )

    except Exception as e:
        logger.error(f"Failed to run model: {e}")
        if args.debug:
            raise
        else:
            console.print(f"[bold red]Error running model:[/bold red] {e}")


def handle_chat(args: argparse.Namespace):
    """Handle chat command."""
    console.print(f"[bold blue]Loading model:[/bold blue] {args.model}")
    try:
        model = create_model(args.model)
        model.initialize()

        console.print("\n[bold green]Starting interactive chat session.[/bold green]")
        console.print("Type 'exit' or 'quit' to end the session.")
        console.print("-" * 40)

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
                    response = model.generate_text(user_input)
                else:
                    response = "Model does not support text generation."

                console.print(f"[bold cyan]AI:[/bold cyan] {response}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        console.print("\n[yellow]Chat session ended.[/yellow]")

    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        if args.debug:
            raise
        else:
            console.print(f"[red]Error starting chat: {e}[/red]")


def handle_info(args: argparse.Namespace):
    """Handle info command."""
    try:
        model = create_model(args.model)

        table = Table(title=f"Model Information: {args.model}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        # Print metadata if available
        if hasattr(model, 'metadata'):
            table.add_row("Name", str(model.metadata.name))
            table.add_row("Version", str(model.metadata.version))
            table.add_row("Author", str(model.metadata.author))
            table.add_row("Type", str(model.metadata.plugin_type.value))
            table.add_row("Description", str(model.metadata.description))

        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            for key, value in info.items():
                if key not in ['name', 'description']:  # Avoid duplication
                    table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)

        # Print docstring if available
        if model.__doc__:
            console.print(
                Panel(
                    model.__doc__.strip(),
                    title="Docstring",
                    border_style="dim"
                )
            )

    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        if args.debug:
            raise
        else:
            console.print(f"[bold red]Error getting info:[/bold red] {e}")


def handle_config(args: argparse.Namespace):
    """Handle config command."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'configs')

    if args.action == 'list':
        console.print(f"Configuration files in {config_dir}:")
        if os.path.exists(config_dir):
            files = [
                f for f in os.listdir(config_dir)
                if f.endswith(('.json', '.yaml', '.yml', '.ini', '.py'))
                and not f.startswith('__')
            ]
            for f in files:
                console.print(f"  - {f}")
        else:
            console.print("[red]Config directory not found.[/red]")

    elif args.action == 'view':
        if not args.file:
            console.print("[red]Error: --file argument is required.[/red]")
            return

        filepath = os.path.join(config_dir, args.file)
        if '..' in args.file or args.file.startswith('/'):
            console.print("[red]Error: Invalid file name.[/red]")
            return

        if os.path.exists(filepath) and os.path.isfile(filepath):
            console.print(f"Content of {args.file}:")
            console.print("-" * 40)
            try:
                with open(filepath, 'r') as f:
                    console.print(f.read())
            except Exception as e:
                console.print(f"[red]Error reading file: {e}[/red]")
            console.print("-" * 40)
        else:
            console.print(f"[red]Error: File {args.file} not found.[/red]")


def handle_benchmark(args: argparse.Namespace):
    """Handle benchmark command."""
    console.print(f"Running benchmark suite: {args.suite}")
    try:
        from inference_pio.benchmarks.scripts.standardized_runner import (
            run_standardized_benchmarks
        )
        run_standardized_benchmarks(benchmark_suite=args.suite)
    except ImportError as e:
        logger.error(f"Failed to import benchmark runner: {e}")
        console.print(f"[red]Error: Could not import benchmark runner. {e}[/red]")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.debug:
            raise
        else:
            console.print(f"[red]Error running benchmark: {e}[/red]")


def handle_check(args: argparse.Namespace):
    """Handle check command."""
    console.print("Performing system check...")
    report = perform_system_check()
    print_system_check_report(report)


def handle_init(args: argparse.Namespace):
    """Handle init command."""
    interactive_init()


def handle_clean(args: argparse.Namespace):
    """Handle clean command."""
    clean_project()


def main():
    """Main entry point for the Inference-PIO CLI."""
    parser = argparse.ArgumentParser(
        description="Inference-PIO: Self-Contained Plugin Architecture"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Inference-PIO {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: list
    parser_list = subparsers.add_parser("list", help="List available models")
    parser_list.set_defaults(func=handle_list)

    # Command: run
    parser_run = subparsers.add_parser("run", help="Run single inference")
    parser_run.add_argument(
        "--model", required=True, help="Model name (e.g., qwen3-0.6b)"
    )
    parser_run.add_argument("--prompt", required=True, help="Input prompt")
    parser_run.set_defaults(func=handle_run)

    # Command: chat
    parser_chat = subparsers.add_parser("chat", help="Start chat session")
    parser_chat.add_argument("--model", required=True, help="Model name")
    parser_chat.set_defaults(func=handle_chat)

    # Command: info
    parser_info = subparsers.add_parser("info", help="Show model information")
    parser_info.add_argument("--model", required=True, help="Model name")
    parser_info.set_defaults(func=handle_info)

    # Command: config
    parser_config = subparsers.add_parser("config", help="Manage configuration")
    parser_config.add_argument(
        "action", choices=["list", "view"], help="Action to perform"
    )
    parser_config.add_argument(
        "--file", help="Config file name (required for view)"
    )
    parser_config.set_defaults(func=handle_config)

    # Command: benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run benchmarks")
    parser_bench.add_argument(
        "--suite",
        default="full",
        choices=["full", "performance", "accuracy"],
        help="Benchmark suite to run"
    )
    parser_bench.set_defaults(func=handle_benchmark)

    # Command: check
    parser_check = subparsers.add_parser(
        "check", help="Check system health and compatibility"
    )
    parser_check.set_defaults(func=handle_check)

    # Command: init
    parser_init = subparsers.add_parser(
        "init", help="Initialize configuration wizard"
    )
    parser_init.set_defaults(func=handle_init)

    # Command: clean
    parser_clean = subparsers.add_parser(
        "clean", help="Clean up temporary files and caches"
    )
    parser_clean.set_defaults(func=handle_clean)

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
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        if args.debug:
            logger.exception("An error occurred:")
        else:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Use --debug for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
