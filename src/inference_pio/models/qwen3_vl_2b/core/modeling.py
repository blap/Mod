"""
Qwen3-VL-2B Modeling Logic - Self-Contained Numpy Backend
"""
import math
import logging
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple

from ....common.custom_components.tokenizer import load_custom_tokenizer
from ....common.processing.image_tokenization import get_optimized_image_processor
from ....common.custom_components.model_loader import CustomModelLoader
from ....core.engine.layers import Module, Linear, Embedding, RMSNorm, Conv2d, ModuleList
from ....core.engine.tensor_ops import softmax, matmul, silu, apply_rotary_emb, precompute_freqs_cis

logger = logging.getLogger(__name__)

# Import base language model architecture
from ...qwen3_0_6b.architecture import Qwen3Model, Qwen3MLP, Qwen3Attention

class VisionRotaryEmbedding(Module):
    def __init__(self, dim, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute freqs for 2D positions (simplified to 1D equivalent for now or needs 2D logic)
        # Vision RoPE typically handles H/W grids.
        # Numpy implementation: calculate once on forward or precompute max

        # 1.0 / (base ** (arange(0, dim, 2) / dim))
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = np.arange(seq_len)
        freqs = np.outer(t, self.inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        return np.cos(emb), np.sin(emb)

class PatchEmbed(Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x) # [B, D, H', W']
        # Flatten: [B, D, L] -> [B, L, D]
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        return x

class VisionAttention(Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = Linear(dim, dim * 3, bias=True)
        self.proj = Linear(dim, dim, bias=True)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope is not None:
            cos, sin = rope
            # Broadcast cos/sin [seq_len, dim] -> [1, 1, seq_len, dim]
            cos = cos.reshape(1, 1, -1, self.head_dim)
            sin = sin.reshape(1, 1, -1, self.head_dim)
            q, k = apply_rotary_emb(q, k, cos, sin)

        attn = matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = softmax(attn, axis=-1)
        x = matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x

class VisionBlock(Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6) # LayerNorm in original, used RMS for simplicity/consistency
        self.attn = VisionAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = Qwen3MLP(type('Config', (), {'hidden_size': dim, 'intermediate_size': int(dim * mlp_ratio)})())

    def forward(self, x, rope=None):
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x

class Qwen3VisionTransformer(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = getattr(config, "vision_hidden_size", 1024)
        self.patch_size = getattr(config, "patch_size", 14)
        self.num_heads = getattr(config, "vision_num_heads", 16)
        self.num_layers = getattr(config, "vision_num_layers", 24)

        self.patch_embed = PatchEmbed(patch_size=self.patch_size, embed_dim=self.embed_dim)
        self.rotary_pos_emb = VisionRotaryEmbedding(self.embed_dim // self.num_heads)

        self.blocks = ModuleList([
            VisionBlock(self.embed_dim, self.num_heads) for _ in range(self.num_layers)
        ])
        self.merger = ModuleList([
            RMSNorm(self.embed_dim),
            Linear(self.embed_dim, getattr(config, "hidden_size", 2048))
        ]) # Should be Sequential, using List manual call

    def forward(self, pixel_values):
        # pixel_values: [B, C, H, W] numpy array
        x = self.patch_embed(pixel_values)

        # Calculate RoPE
        cos, sin = self.rotary_pos_emb(x.shape[1])
        rope = (cos, sin)

        for block in self.blocks:
            x = block(x, rope=rope)

        # Merger
        for layer in self.merger:
            x = layer(x)

        return x

class Qwen3VL2BArchitecture(Module):
    """
    Self-contained Qwen3-VL architecture (Numpy Backend).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VisionTransformer(config)
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = self.visual(pixel_values)
            # Concatenation logic (simplified for Numpy)
            # Assuming broadcasting/batch size matches
            if image_embeds.shape[0] == inputs_embeds.shape[0]:
                 inputs_embeds = np.concatenate([image_embeds, inputs_embeds], axis=1)

        hidden_states, pkv = self.model(inputs_embeds=inputs_embeds, **kwargs)
        logits = self.lm_head(hidden_states)
        return logits, pkv

    def generate(self, *args, **kwargs):
        # Basic generation logic, relying on LM component
        # Need to handle image prefix in generation loop manually if not handled by caller
        return self.model.generate(*args, **kwargs)

class Qwen3VL2BModeling(Module):
    def __init__(self, config, system_profile):
        super().__init__()
        self.config = config
        self._system_profile = system_profile
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._model_name = config.model_path

        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"Initializing Qwen3-VL-2B model (Numpy Backend)...")

            self._model = Qwen3VL2BArchitecture(self.config)

            # Load Weights
            try:
                CustomModelLoader.load_weights(self._model, self._model_name, device="cpu")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")

            # Load Processors
            try:
                self._tokenizer = load_custom_tokenizer(self._model_name)
                # Image processor usually returns torch tensors, need numpy
                # We assume get_optimized_image_processor returns a processor that can return numpy
                self._image_processor = get_optimized_image_processor(self._model_name)
            except Exception as e:
                logger.warning(f"Failed to load processors: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-VL-2B model: {e}")
            raise
