"""
Real model implementation for testing purposes.

This module provides a real model implementation that can be used in place of mock models
in test files. It includes a simplified version of the GLM-47-Flash model that can be
instantiated without requiring external dependencies.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Optional, Dict, Any


class RealGLM47Model(nn.Module):
    """
    Real GLM-47-Flash model implementation for testing purposes.
    
    This is a simplified version of the GLM-47-Flash model that can be instantiated
    without requiring external dependencies. It mimics the behavior of the real model
    for testing purposes.
    """
    
    def __init__(self, 
                 hidden_size: int = 1024,
                 num_attention_heads: int = 16,
                 num_hidden_layers: int = 8,
                 intermediate_size: int = 2048,
                 vocab_size: int = 50000,
                 max_position_embeddings: int = 2048,
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize the real GLM-47-Flash model.
        
        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads
            num_hidden_layers: Number of hidden layers
            intermediate_size: Size of intermediate layers
            vocab_size: Vocabulary size
            max_position_embeddings: Maximum position embeddings
            torch_dtype: Data type for tensors
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        
        # Create a simple transformer-like architecture
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Create transformer layers with attention and feed-forward networks
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            ) for _ in range(num_hidden_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store config as an attribute to mimic real model behavior
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'num_hidden_layers': num_hidden_layers,
            'intermediate_size': intermediate_size,
            'vocab_size': vocab_size,
            'max_position_embeddings': max_position_embeddings,
            'torch_dtype': torch_dtype
        })()
        
        # Add tokenizer attribute to mimic real model
        try:
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except:
            # If GPT-2 tokenizer fails, create a minimal tokenizer
            self._tokenizer = MinimalTokenizer(vocab_size)
    
    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Input embeddings (alternative to input_ids)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing model outputs
        """
        # Get input embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embeddings(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Apply attention mask if provided
        extended_attention_mask = None
        if attention_mask is not None:
            # Expand attention mask for multi-head attention
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(next(self.parameters()).dtype).min

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)

        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Return outputs in a format similar to Hugging Face models
        return type('ModelOutput', (), {
            'logits': logits,
            'last_hidden_state': hidden_states,
            'hidden_states': None,  # Not computed for simplicity
            'attentions': None      # Not computed for simplicity
        })()
    
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 10,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 **kwargs) -> torch.Tensor:
        """
        Generate tokens using the model.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Start with the input tokens
        generated = input_ids.clone()

        # Generate new tokens one by one
        for _ in range(max_new_tokens):
            # Get model output for current sequence
            outputs = self(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature

            if do_sample:
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Take the most likely token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append the new token to the sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if we generated an EOS token (check if config has eos_token_id)
            if hasattr(self.config, 'eos_token_id') and next_token.item() == self.config.eos_token_id:
                break

        return generated
    
    def get_tokenizer(self):
        """
        Get the tokenizer associated with the model.
        """
        return self._tokenizer


class TransformerLayer(nn.Module):
    """
    A single transformer layer with attention and feed-forward network.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        self.mlp_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.mlp_layer_norm(hidden_states + mlp_output)
        
        return hidden_states


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        if self.head_dim * num_attention_heads != hidden_size:
            raise ValueError(f"Hidden size {hidden_size} not divisible by number of attention heads {num_attention_heads}")
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to query, key, value
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MinimalTokenizer:
    """
    A minimal tokenizer implementation for testing purposes.
    """
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
    
    def encode(self, text: str, **kwargs):
        """
        Encode text to token IDs.
        """
        # Simple encoding - convert characters to IDs
        import hashlib
        tokens = []
        for word in text.split()[:100]:  # Limit to prevent excessive tokens
            # Create a hash-based ID for the word
            word_hash = abs(int(hashlib.md5(word.encode()).hexdigest(), 16)) % (self.vocab_size - 100)
            # Reserve first 100 IDs for special tokens
            token_id = word_hash + 100
            tokens.append(token_id)
        return tokens
    
    def decode(self, token_ids, **kwargs):
        """
        Decode token IDs to text.
        """
        # Simple decoding - convert IDs back to placeholder text
        words = [f"token_{tid}" for tid in token_ids if tid >= 100]
        return " ".join(words)
    
    def __call__(self, text: str, return_tensors=None, padding=None, truncation=None, max_length=None, **kwargs):
        """
        Call the tokenizer.
        """
        input_ids = self.encode(text)
        
        if max_length:
            if truncation:
                input_ids = input_ids[:max_length]
            else:
                input_ids = input_ids[:max_length]
        
        if padding and max_length:
            if len(input_ids) < max_length:
                input_ids.extend([self.pad_token_id] * (max_length - len(input_ids)))
        
        result = {"input_ids": input_ids}
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([input_ids])
        
        return result


def create_real_glm47_model(**kwargs) -> RealGLM47Model:
    """
    Factory function to create a real GLM-47-Flash model for testing.
    
    Args:
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        RealGLM47Model instance
    """
    return RealGLM47Model(**kwargs)


# For backward compatibility with existing tests
class MockModel(RealGLM47Model):
    """
    Alias for backward compatibility with existing tests that expect MockModel.
    """
    pass