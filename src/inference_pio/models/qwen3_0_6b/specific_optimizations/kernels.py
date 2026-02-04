import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class OptimizedRMSNorm(nn.Module):
    """
    Optimized RMSNorm implementation using torch.compile-friendly operations
    or custom Triton kernels if available in the future.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Prefer float32 for stability in norm
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

def apply_qwen_kernels(model: nn.Module) -> nn.Module:
    """
    Replaces standard modules with optimized versions specific to Qwen3.
    """
    logger.info("Applying Qwen3-specific optimized kernels...")

    replacements = 0

    # Example: Replace standard LayerNorm/RMSNorm with our optimized version
    # Note: Accessing private members like _modules is necessary for in-place replacement
    for name, module in model.named_children():
        if "norm" in name.lower() and isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # Check compatibility
            if hasattr(module, 'normalized_shape') and len(module.normalized_shape) == 1:
                dim = module.normalized_shape[0]
                eps = module.eps

                # Create optimized replacement
                opt_norm = OptimizedRMSNorm(dim, eps)
                opt_norm.weight.data = module.weight.data.clone()

                setattr(model, name, opt_norm)
                replacements += 1
        else:
            # Recursively apply
            apply_qwen_kernels(module)

    if replacements > 0:
        logger.info(f"Replaced {replacements} normalization layers with OptimizedRMSNorm")

    return model
