import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class Qwen3CoderNextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class Qwen3CoderNextGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")


def apply_qwen3_coder_next_optimizations(model: nn.Module) -> nn.Module:
    logger.info("Applying Qwen3-Coder-Next specific optimizations...")

    replacements = {"norm": 0, "act": 0}

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if (
                hasattr(module, "normalized_shape")
                and len(module.normalized_shape) == 1
            ):
                dim = module.normalized_shape[0]
                eps = module.eps

                parent_name, child_name = (
                    name.rsplit(".", 1) if "." in name else (None, name)
                )

                if parent_name:
                    parent_module = _get_parent_module(model, parent_name)
                    opt_norm = Qwen3CoderNextRMSNorm(dim, eps)
                    if hasattr(module, "weight") and module.weight is not None:
                        opt_norm.weight.data.copy_(module.weight.data)

                    setattr(parent_module, child_name, opt_norm)
                    replacements["norm"] += 1

        if isinstance(module, nn.GELU):
            parent_name, child_name = (
                name.rsplit(".", 1) if "." in name else (None, name)
            )
            if parent_name:
                parent_module = _get_parent_module(model, parent_name)
                setattr(parent_module, child_name, Qwen3CoderNextGELU())
                replacements["act"] += 1

    logger.info(f"Qwen3-Coder-Next optimizations applied: {replacements}")
    return model


def _get_parent_module(model, parent_name):
    parent = model
    for n in parent_name.split("."):
        if n:
            parent = getattr(parent, n)
    return parent
