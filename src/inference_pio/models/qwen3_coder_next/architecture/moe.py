"""
Mixture of Experts (MoE) Module for Qwen3-Coder-Next
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3CoderNextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.expert_intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen3CoderNextMoE(nn.Module):
    """
    Mixture of Experts
    - 512 Experts
    - 10 Activated
    - 1 Shared Expert
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_activated = config.num_activated_experts

        # Experts
        self.experts = nn.ModuleList([Qwen3CoderNextMLP(config) for _ in range(self.num_experts)])

        # Shared Expert (Always active)
        self.shared_expert = Qwen3CoderNextMLP(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        # Router
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router Logits
        router_logits = self.gate(hidden_states_flat)

        # Top-K Gating
        routing_weights, selected_experts = torch.topk(router_logits, self.num_activated, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # Expert Computation
        # Naive implementation for structure; optimized version should use sparse dispatch kernels
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Shared Expert
        shared_output = self.shared_expert(hidden_states_flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states_flat))
        final_hidden_states += shared_output * shared_gate

        # Dynamic Experts
        # Loop is inefficient, optimized implementation needs efficient gather/scatter
        for i in range(self.num_activated):
            expert_idx = selected_experts[:, i]
            weight = routing_weights[:, i].unsqueeze(1)

            # This part is extremely slow in Python loop for 512 experts
            # In real execution, we group tokens by expert
            for expert_id in range(self.num_experts):
                 mask = (expert_idx == expert_id)
                 if mask.any():
                     tokens = hidden_states_flat[mask]
                     out = self.experts[expert_id](tokens)
                     final_hidden_states[mask] += out * weight[mask]

        return final_hidden_states.view(bsz, seq_len, hidden_dim)
