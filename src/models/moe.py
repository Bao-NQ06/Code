"""
Mixture of Experts (MoE) module for Whisper fine-tuning.

Replaces selected FFN layers in Whisper decoder (or encoder)
with MoE layers that route tokens to top-k experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ExpertFFN(nn.Module):
    """A single expert: cloned from original Whisper FFN block."""

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, activation_fn):
        super().__init__()
        self.fc1 = copy.deepcopy(fc1)
        self.fc2 = copy.deepcopy(fc2)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class TopKRouter(nn.Module):
    """Learned router that selects top-k experts per token."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            indices: (batch, seq_len, top_k) - expert indices
            weights: (batch, seq_len, top_k) - gating weights
            router_logits: (batch, seq_len, num_experts) - for aux loss
        """
        logits = self.gate(x)  # (B, T, num_experts)
        weights, indices = torch.topk(
            F.softmax(logits, dim=-1), self.top_k, dim=-1
        )
        # Renormalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return indices, weights, logits


class MoELayer(nn.Module):
    """Mixture of Experts layer replacing a single FFN block."""

    def __init__(
        self,
        fc1: nn.Linear,
        fc2: nn.Linear,
        activation_fn,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        d_model = fc1.in_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(fc1, fc2, activation_fn)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: load balancing auxiliary loss
        """
        B, T, D = x.shape
        indices, weights, router_logits = self.router(x)

        # Compute output via weighted expert combination
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # (B, T)
            expert_weight = weights[:, :, k].unsqueeze(-1)  # (B, T, 1)

            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)  # (B, T)
                if mask.any():
                    expert_input = x[mask]  # (N, D)
                    expert_out = self.experts[e_id](expert_input)
                    output[mask] += (expert_weight[mask] * expert_out)

        aux_loss = self._load_balance_loss(router_logits)
        return output, aux_loss

    def _load_balance_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        probs = F.softmax(router_logits, dim=-1)  # (B, T, E)
        avg_probs = probs.mean(dim=[0, 1])  # (E,)
        uniform = torch.ones_like(avg_probs) / self.num_experts
        return F.mse_loss(avg_probs, uniform) * self.num_experts


def inject_moe_into_whisper(model, moe_cfg: dict):
    """Replace selected FFN layers in Whisper with MoE layers.

    Args:
        model: WhisperForConditionalGeneration
        moe_cfg: MoE configuration dict

    Returns:
        Modified model with MoE layers injected.
    """
    num_experts = moe_cfg.get("num_experts", 4)
    top_k = moe_cfg.get("top_k", 2)
    capacity_factor = moe_cfg.get("capacity_factor", 1.25)
    apply_to = moe_cfg.get("apply_to", "decoder")
    replace_every_n = moe_cfg.get("replace_every_n", 2)

    targets = []
    if apply_to in ("decoder", "both"):
        targets.append(("decoder", model.model.decoder.layers))
    if apply_to == "both":
        targets.append(("encoder", model.model.encoder.layers))

    replaced = 0
    for name, layers in targets:
        for i, layer in enumerate(layers):
            if i % replace_every_n != 0:
                continue
            activation_fn = layer.activation_fn
            moe_layer = MoELayer(
                fc1=layer.fc1,
                fc2=layer.fc2,
                activation_fn=activation_fn,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
            )
            layer._moe_layer = moe_layer
            _patch_layer_forward(layer, moe_layer)
            replaced += 1

    print(f"MoE: Replaced {replaced} FFN layers with {num_experts} experts")
    return model


def _patch_layer_forward(layer, moe_layer: MoELayer):
    """Monkey-patch the layer's forward to route through MoE."""
    original_forward = layer.forward

    def moe_forward(*args, **kwargs):
        # Run original forward up to the FFN (attention part)
        # Then replace FFN output with MoE output
        hidden_states = args[0] if args else kwargs.get("hidden_states")

        # Self-attention
        residual = hidden_states
        hidden_states = layer.self_attn_layer_norm(hidden_states)
        attn_out = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=kwargs.get("attention_mask"),
            layer_head_mask=kwargs.get("layer_head_mask"),
            output_attentions=kwargs.get("output_attentions", False),
        )
        hidden_states = residual + attn_out[0]

        # Cross-attention (decoder only)
        if hasattr(layer, "encoder_attn"):
            residual = hidden_states
            hidden_states = layer.encoder_attn_layer_norm(hidden_states)
            cross_out = layer.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=kwargs.get("encoder_hidden_states"),
                attention_mask=kwargs.get("cross_attn_layer_head_mask"),
                output_attentions=kwargs.get("output_attentions", False),
            )
            hidden_states = residual + cross_out[0]

        # MoE FFN (replaces original FFN)
        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        moe_out, aux_loss = moe_layer(hidden_states)
        hidden_states = residual + moe_out

        # Store aux_loss on the layer for retrieval during training
        layer._moe_aux_loss = aux_loss

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (attn_out[1],)
        return outputs

    layer.forward = moe_forward
