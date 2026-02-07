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

class OptimizedGELU(nn.Module):
    """
    Optimized GELU activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
         return torch.nn.functional.gelu(x, approximate='tanh')


def apply_qwen_kernels(model: nn.Module) -> nn.Module:
    """
    Replaces standard modules with optimized versions specific to Qwen3.
    """
    logger.info("Applying Qwen3-specific optimized kernels...")

    replacements = {"norm": 0, "act": 0}

    # We iterate over named_modules to find replacements.
    # Note: modifying the model structure while iterating over it via named_modules is generally unsafe
    # if we were removing modules, but here we are replacing them in-place via parent.
    # We collect modifications and apply them to avoid iteration issues if any.
    modifications = []

    for name, module in model.named_modules():
        # Check for LayerNorm/RMSNorm replacement
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
             if hasattr(module, 'normalized_shape') and len(module.normalized_shape) == 1:
                modifications.append((name, module, "norm"))

        # Check for GELU replacement
        if isinstance(module, nn.GELU):
             modifications.append((name, module, "act"))

    for name, module, type_tag in modifications:
        parent_name, child_name = name.rsplit(".", 1) if "." in name else (None, name)

        if parent_name:
            parent_module = _get_parent_module(model, parent_name)
        else:
            parent_module = model

        if type_tag == "norm":
            dim = module.normalized_shape[0]
            eps = module.eps
            opt_norm = OptimizedRMSNorm(dim, eps)
            if hasattr(module, 'weight') and module.weight is not None:
                opt_norm.weight.data.copy_(module.weight.data)
            setattr(parent_module, child_name, opt_norm)
            replacements["norm"] += 1

        elif type_tag == "act":
            setattr(parent_module, child_name, OptimizedGELU())
            replacements["act"] += 1

    logger.info(f"Replaced {replacements} layers with optimized versions")
    return model

def _get_parent_module(model: nn.Module, parent_name: str) -> nn.Module:
    parent_module = model
    for n in parent_name.split("."):
        if n:
            parent_module = getattr(parent_module, n)
    return parent_module
