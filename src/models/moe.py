"""
Mixture of Experts (MoE) for Whisper fine-tuning.

Replaces selected FFN blocks with sparse Top-K Gated MoE layers.
Expert dispatch is vectorized over the (batch × time) dimension to
minimize Python-level loops and maximize GPU utilization.

Load-balancing auxiliary loss encourages uniform expert utilization.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Single expert: an independent copy of the Whisper FFN block."""

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, activation_fn):
        super().__init__()
        self.fc1 = copy.deepcopy(fc1)
        self.fc2 = copy.deepcopy(fc2)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(x)))


class TopKRouter(nn.Module):
    """Learned linear router selecting top-k experts per token."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, D) flat token representations
        Returns:
            indices: (N, top_k)
            weights: (N, top_k)  renormalized softmax weights
            logits:  (N, E)      raw gating logits (for aux loss)
        """
        logits = self.gate(x)                                     # (N, E)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)  # (N, k)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return indices, weights, logits


class MoELayer(nn.Module):
    """
    Sparse Top-K MoE replacing a single Whisper FFN block.

    Dispatch strategy (vectorized):
      - Flatten (B, T, D) → (N, D)  where N = B*T
      - For each position in top_k, scatter-gather tokens to each expert
      - Accumulate weighted expert outputs
      - Reshape back to (B, T, D)

    This avoids nested Python loops over the batch dimension, letting
    PyTorch/CUDA execute one kernel call per expert per top-k slot.
    """

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, activation_fn,
                 num_experts: int = 8, top_k: int = 2,
                 capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = TopKRouter(fc1.in_features, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(fc1, fc2, activation_fn) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            output:   (B, T, D)
            aux_loss: scalar load-balance loss
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)                # (N, D), N = B*T
        indices, weights, logits = self.router(flat)  # (N, k), (N, k), (N, E)

        output = torch.zeros_like(flat)        # (N, D)

        for k in range(self.top_k):
            slot_idx = indices[:, k]           # (N,)  — which expert per token
            slot_w = weights[:, k:k+1]        # (N, 1)

            for e_id in range(self.num_experts):
                mask = slot_idx == e_id        # (N,) bool
                if not mask.any():
                    continue
                out = self.experts[e_id](flat[mask])    # (M, D)
                output[mask] += slot_w[mask] * out

        return output.reshape(B, T, D), self._load_balance_loss(logits)

    def _load_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss that pushes average routing probabilities toward uniform."""
        probs = F.softmax(logits, dim=-1)           # (N, E)
        avg = probs.mean(dim=0)                     # (E,)
        target = torch.full_like(avg, 1.0 / self.num_experts)
        return F.mse_loss(avg, target) * self.num_experts


def inject_moe_into_whisper(model, moe_cfg: dict):
    """Replace every N-th FFN block in the target part of Whisper with MoE."""
    num_experts = moe_cfg.get("num_experts", 8)
    top_k = moe_cfg.get("top_k", 2)
    capacity_factor = moe_cfg.get("capacity_factor", 1.25)
    apply_to = moe_cfg.get("apply_to", "decoder")
    every_n = moe_cfg.get("replace_every_n", 2)

    targets = []
    if apply_to in ("decoder", "both"):
        targets.append(model.model.decoder.layers)
    if apply_to in ("encoder", "both"):
        targets.append(model.model.encoder.layers)

    replaced = 0
    for layers in targets:
        for i, layer in enumerate(layers):
            if i % every_n != 0:
                continue
            moe = MoELayer(
                fc1=layer.fc1,
                fc2=layer.fc2,
                activation_fn=layer.activation_fn,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
            )
            layer._moe_layer = moe
            _patch_ffn(layer, moe)
            replaced += 1

    print(f"  MoE: replaced {replaced} FFN blocks ({num_experts} experts, top-{top_k})")
    return model


def _patch_ffn(layer, moe: MoELayer) -> None:
    """Monkey-patch layer.forward to route through MoE instead of original FFN."""

    def moe_forward(*args, **kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]

        # Self-attention with residual
        residual = hidden
        hidden = layer.self_attn_layer_norm(hidden)
        attn_out = layer.self_attn(
            hidden_states=hidden,
            attention_mask=kwargs.get("attention_mask"),
            layer_head_mask=kwargs.get("layer_head_mask"),
            output_attentions=kwargs.get("output_attentions", False),
        )
        hidden = residual + attn_out[0]

        # Cross-attention with residual (decoder only)
        if hasattr(layer, "encoder_attn"):
            residual = hidden
            hidden = layer.encoder_attn_layer_norm(hidden)
            cross_out = layer.encoder_attn(
                hidden_states=hidden,
                key_value_states=kwargs.get("encoder_hidden_states"),
                attention_mask=kwargs.get("cross_attn_layer_head_mask"),
                output_attentions=kwargs.get("output_attentions", False),
            )
            hidden = residual + cross_out[0]

        # MoE FFN with residual
        residual = hidden
        hidden = layer.final_layer_norm(hidden)
        moe_out, aux_loss = moe(hidden)
        hidden = residual + moe_out
        layer._moe_aux_loss = aux_loss

        result = (hidden,)
        if kwargs.get("output_attentions", False):
            result += (attn_out[1],)
        return result

    layer.forward = moe_forward
