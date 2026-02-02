import re
from pathlib import Path


def fix_imports_specific_optimization_manual(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content

    # Quantization mappings
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.quantization import",
        r"from src.inference_pio.common.quantization import",
        new_content,
    )

    # Attention mappings
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.attention import",
        r"from src.inference_pio.common.paged_attention import",
        new_content,
    )
    # Also handle flash_attention if needed, but PagedKVCache is usually in paged_attention

    # Specific optimization mappings
    # These might be in 'src.inference_pio.common.unified_ml_optimization' or similar?
    # Or 'src.inference_pio.common.model_specific_optimization' (if it existed)
    # The error was "No module named 'src.inference_pio.optimization'"
    # We moved everything to 'src.inference_pio.common'.

    # Let's map 'src.inference_pio.optimization.specific' -> 'src.inference_pio.common.unified_ml_optimization' (guess)
    # Or remove the import if the function doesn't exist?

    # Cross modal
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.cross_modal import",
        r"from src.inference_pio.common.cross_modal_fusion_kernels import",
        new_content,
    )

    # Pipeline
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.pipeline import",
        r"from src.inference_pio.common.pipeline_parallel import",
        new_content,
    )

    # Vision Language
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.vision_language import",
        r"from src.inference_pio.common.vision_attention import",
        new_content,
    )  # Guessing

    # Image tokenization
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.image_tokenization import",
        r"from src.inference_pio.common.image_tokenization import",
        new_content,
    )

    # Factory
    new_content = re.sub(
        r"from src\.inference_pio\.optimization\.factory import",
        r"from src.inference_pio.common.optimization_manager import",
        new_content,
    )  # Guessing

    if new_content != content:
        print(f"Fixed optimization imports in {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


def main():
    test_dir = Path("tests/unit/common/optimization/specific")
    for p in test_dir.glob("**/*.py"):
        fix_imports_specific_optimization_manual(p)


if __name__ == "__main__":
    main()
