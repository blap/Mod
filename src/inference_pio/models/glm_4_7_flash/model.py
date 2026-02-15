"""
GLM-4.7-Flash Model Implementation - Self-Contained
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from ...core.engine.backend import Module, Tensor, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention, cat
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer, CustomBPETokenizer
from .config import GLM47FlashConfig

logger = logging.getLogger(__name__)

class GLM47FlashModel(Module):
    def __init__(self, config: GLM47FlashConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []

        dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = precompute_freqs_cis(dim, config.max_position_embeddings)

        # Config uses num_hidden_layers
        for i in range(config.num_hidden_layers):
            l = GLM47FlashLayer(config, self.cos, self.sin)
            self.layers.append(l)
            self._modules[f"layer_{i}"] = l

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.scheduler = None
        self._tokenizer = None

        self._initialize_model()

    def _resolve_model_path(self) -> str:
        """
        Resolves the model path with the following priority:
        1. H:/GLM-4.7-Flash
        2. Local Cache (src/inference_pio/models/glm_4_7_flash/_model_cache/GLM-4.7-Flash)
        3. Download from HuggingFace
        """
        model_name = "GLM-4.7-Flash"
        hf_repo = "zai-org/GLM-4.7-Flash"

        # 1. Check H: Drive
        h_drive_path = os.path.join("H:/", model_name)
        if os.path.exists(h_drive_path):
            logger.info(f"Found model on H: drive: {h_drive_path}")
            return h_drive_path

        # 2. Check Local Cache
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "_model_cache")
        local_path = os.path.join(cache_dir, model_name)

        if os.path.exists(local_path):
            if os.listdir(local_path):
                logger.info(f"Found model in local cache: {local_path}")
                return local_path

        # 3. Download
        logger.info(f"Model not found. Attempting to download {hf_repo} to {local_path}...")

        os.makedirs(cache_dir, exist_ok=True)

        # Check Disk Space (need approx 2GB for small models, 7GB+ for larger)
        total, used, free = shutil.disk_usage(cache_dir)
        required_space = 2 * 1024 * 1024 * 1024 # 2GB Safety

        if free < required_space:
            raise RuntimeError(f"Insufficient disk space. Required: 2GB, Available: {free/1024/1024/1024:.2f}GB")

        try:
            logger.info("Cloning from HuggingFace...")
            subprocess.run(
                ["git", "clone", f"https://huggingface.co/{hf_repo}", local_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Download complete.")
            return local_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            raise RuntimeError(f"Failed to download model from {hf_repo}")

    def _initialize_model(self):
        logger.info("Initializing GLM-4.7-Flash model...")

        try:
            model_path = self._resolve_model_path()
            CustomModelLoader.load_weights(self, model_path, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Model will use random initialization.")

        try:
            model_path = self._resolve_model_path()
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Text processing will be limited.")

    def forward(self, input_ids: Tensor, past_key_values: Optional[List[Tensor]] = None, use_cache: bool = False, cache_position: int = 0):
        h = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
             past_key_values = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)

            target_device = layer.input_layernorm.weight.device
            if h.device != target_device:
                h = h.to(target_device)

            pkv = past_key_values[i] if past_key_values else None
            h = layer(h, past_key_value=pkv, use_cache=use_cache, cache_position=cache_position)

        h = self.final_layernorm(h)
        return h

    def generate(self, input_ids: Tensor, max_new_tokens: int = 10):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_seq_len = seq_len + max_new_tokens

        device = input_ids.device
        # GQA Cache Allocation: [Batch, MaxSeq, KV_Heads, HeadDim]
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        past_key_values = []

        for _ in range(len(self.layers)):
             # GQA: Use num_key_value_heads
             k_cache = Tensor([batch_size, max_seq_len, self.config.num_key_value_heads, head_dim], device=device)
             v_cache = Tensor([batch_size, max_seq_len, self.config.num_key_value_heads, head_dim], device=device)
             k_cache.fill(0.0)
             v_cache.fill(0.0)
             past_key_values.append((k_cache, v_cache))

        current_ids = input_ids
        for step in range(max_new_tokens):
            curr_seq_len = current_ids.shape[1]
            if step == 0:
                model_input = current_ids
                cache_position = 0
            else:
                model_input = current_ids.slice([0, curr_seq_len-1], [batch_size, 1])
                cache_position = curr_seq_len - 1

            h = self.forward(model_input, past_key_values=past_key_values, use_cache=True, cache_position=cache_position)
            logits = self.lm_head(h)

            vocab_size = logits.shape[2]
            last_logits = logits.slice([0, logits.shape[1]-1, 0], [batch_size, 1, vocab_size])
            next_token = last_logits.argmax()

            current_ids = cat([current_ids, next_token], axis=1)
        return current_ids

class GLM47FlashLayer(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention = GLM47FlashAttention(config, cos, sin)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GLM47FlashMLP(config)

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        h = self.input_layernorm(x)
        h = self.self_attention(h, past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position)
        x = x + h
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        return x + h

class GLM47FlashAttention(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        # GQA Projection Size: (Heads + 2 * KV_Heads) * HeadDim
        # Example: 2048 hidden, 16 heads -> 128 dim
        # KV heads = 8 -> 8 * 128 = 1024
        # Total QKV dim = 2048 + 1024 + 1024 = 4096
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.total_proj_size = self.q_size + 2 * self.kv_size

        self.query_key_value = Linear(self.hidden_size, self.total_proj_size, bias=True)
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cos = cos
        self.sin = sin
        self.scale = self.head_dim ** -0.5
        self.group_size = self.num_heads // self.num_kv_heads

    def repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """
        Naive repeat_kv via concat (Slow but functional without custom repeat kernel).
        x: [B, S, n_kv_heads, head_dim] -> [B, S, n_heads, head_dim]
        """
        if n_rep == 1:
            return x

        # To repeat heads, we can't easily use 'cat' on dim 2 directly in a loop efficiently in python.
        # But if backend `cat` supports list, we can.
        # Or if backend supports `broadcast` or `expand` (not implemented).
        # We will iterate over heads and duplicate them.
        # Ideally, we should implement `tensor_repeat_interleave` in backend.
        # Without it, we might rely on broadcasting in attention if supported.
        # My backend `scaled_dot_product_attention` does `q.matmul(k.T)`.
        # This implies standard attention where heads match.

        # Workaround: Use Python loop to construct list of tensors and cat them.
        # This is extremely slow for large sequences but functional.
        B, S, H_kv, D = x.shape

        # We need to interleave repeats: [h1, h1, h2, h2...]
        # Backend `cat` concatenates a list.
        # Extract each head:
        head_tensors = []
        for h in range(H_kv):
            # Slice head h: [B, S, 1, D]
            head = x.slice([0, 0, h, 0], [B, S, 1, D])
            for _ in range(n_rep):
                head_tensors.append(head)

        # Cat along head dim (axis 2)
        out = cat(head_tensors, axis=2)
        return out

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        # QKV Proj
        qkv = self.query_key_value(x) # [B, S, Total_Proj]

        B, Seq, _ = qkv.shape

        # Split Q, K, V
        # Q: [0 : q_size]
        # K: [q_size : q_size + kv_size]
        # V: [q_size + kv_size : total]

        q = qkv.slice([0, 0, 0], [B, Seq, self.q_size])
        k = qkv.slice([0, 0, self.q_size], [B, Seq, self.kv_size])
        v = qkv.slice([0, 0, self.q_size + self.kv_size], [B, Seq, self.kv_size])

        # Reshape to Heads
        q = q.reshape([B, Seq, self.num_heads, self.head_dim])
        k = k.reshape([B, Seq, self.num_kv_heads, self.head_dim])
        v = v.reshape([B, Seq, self.num_kv_heads, self.head_dim])

        # RoPE (Apply to K and Q)
        # Slicing cos/sin for current position
        start = [cache_position, 0]
        shape = [Seq, self.cos.shape[1]]
        c = self.cos.slice(start, shape)
        s = self.sin.slice(start, shape)

        q, k = q.rope(k, c, s)

        # KV Cache Update
        if use_cache and past_key_value is not None:
             k_cache, v_cache = past_key_value
             start_indices = [0, cache_position, 0, 0]
             k_cache.set_slice(k, start_indices)
             v_cache.set_slice(v, start_indices)

             valid_len = cache_position + Seq
             k = k_cache.slice([0,0,0,0], [B, valid_len, self.num_kv_heads, self.head_dim])
             v = v_cache.slice([0,0,0,0], [B, valid_len, self.num_kv_heads, self.head_dim])

        # GQA Repeat
        if self.group_size > 1:
            k = self.repeat_kv(k, self.group_size)
            v = self.repeat_kv(v, self.group_size)

        # Attention
        out = scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Flatten
        out = out.reshape([B, Seq, self.hidden_size])

        return self.dense(out)

class GLM47FlashMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.dense_4h_to_h = Linear(config.hidden_size * 4, config.hidden_size, bias=False)

    def forward(self, x):
        h = self.dense_h_to_4h(x)
        h = h.gelu()
        return self.dense_4h_to_h(h)

__all__ = ["GLM47FlashModel"]
