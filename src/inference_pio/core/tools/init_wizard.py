import os
import yaml
from rich.prompt import Prompt, Confirm
from .rich_utils import console


def interactive_init():
    """Run the interactive initialization wizard."""
    console.print("[bold cyan]Inference-PIO Configuration Wizard[/bold cyan]")
    console.print("This wizard will help you create a 'user_config.yaml' file.")

    config = {}

    # 1. Default Model
    config['default_model'] = Prompt.ask(
        "Default model to load",
        default="qwen3-0.6b",
        choices=[
            "qwen3-0.6b", "qwen3-vl-2b", "glm-4-7-flash",
            "qwen3-4b", "qwen3-coder-30b"
        ]
    )

    # 2. Hardware Settings
    config['hardware'] = {}
    config['hardware']['use_gpu'] = Confirm.ask(
        "Use GPU if available?", default=True
    )

    if config['hardware']['use_gpu']:
        config['hardware']['gpu_layers'] = int(
            Prompt.ask("GPU layers to offload", default="32")
        )

    # 3. Memory Settings
    config['memory'] = {}
    config['memory']['enable_disk_offloading'] = Confirm.ask(
        "Enable disk offloading for low RAM?", default=True
    )
    if config['memory']['enable_disk_offloading']:
        config['memory']['offload_dir'] = Prompt.ask(
            "Offload directory", default="./offload_cache"
        )

    # 4. Storage
    # Detect H: drive again to suggest default
    default_model_dir = "models"
    import platform
    if platform.system() == 'Windows' and os.path.exists("H:/"):
        default_model_dir = "H:/models"
    elif os.path.exists("/mnt/h"):
        default_model_dir = "/mnt/h/models"

    config['storage'] = {}
    config['storage']['models_dir'] = Prompt.ask(
        "Directory to store/load models", default=default_model_dir
    )

    # Save
    output_file = "user_config.yaml"
    if os.path.exists(output_file):
        if not Confirm.ask(
            f"File {output_file} exists. Overwrite?", default=False
        ):
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        console.print(
            f"[bold green]Configuration saved to {output_file}[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Failed to save config: {e}[/bold red]")
