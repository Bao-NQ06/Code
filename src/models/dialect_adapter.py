"""
Dialect-Adaptive Auxiliary Task (DAAT) — Novel Research Contribution
=====================================================================
Research motivation:
    Vietnamese ASR errors are not uniformly distributed across regions.
    Central dialects suffer disproportionately because a standard ASR
    encoder never learns explicit dialect-discriminative features.

Approach — Multi-task Learning:
    A lightweight MLP classifier head is attached to the pooled encoder
    output. During fine-tuning:

        L_total = L_ASR  +  λ · L_dialect_CE

    The dialect gradient flows back into the LoRA-adapted encoder layers,
    forcing them to produce more dialect-discriminative representations.
    At inference the classifier head is discarded — zero overhead.

Side benefit: Free dialect detector for metadata enrichment and
per-region WER attribution in the evaluation report.

Dialect mapping (3 classes):
    0 — Northern  (Bắc)
    1 — Central   (Trung)  ← historically hardest for ASR
    2 — Southern  (Nam)
   -1 — Unknown  (excluded from aux loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DIALECT_TO_IDX: dict[str, int] = {
    "Northern": 0,
    "Central": 1,
    "Southern": 2,
}
IDX_TO_DIALECT: dict[int, str] = {v: k for k, v in DIALECT_TO_IDX.items()}

# Province → dialect index look-up table (province substring → index)
_PROVINCE_MAP: list[tuple[str, int]] = [
    # Northern (0)
    ("hà nội", 0), ("hải phòng", 0), ("quảng ninh", 0), ("bắc giang", 0),
    ("bắc kạn", 0), ("bắc ninh", 0), ("cao bằng", 0), ("điện biên", 0),
    ("hà giang", 0), ("hà nam", 0), ("hải dương", 0), ("hòa bình", 0),
    ("hưng yên", 0), ("lai châu", 0), ("lạng sơn", 0), ("lào cai", 0),
    ("nam định", 0), ("ninh bình", 0), ("phú thọ", 0), ("sơn la", 0),
    ("thái bình", 0), ("thái nguyên", 0), ("tuyên quang", 0),
    ("vĩnh phúc", 0), ("yên bái", 0),
    # Central (1)
    ("đà nẵng", 1), ("huế", 1), ("thừa thiên", 1), ("quảng bình", 1),
    ("quảng trị", 1), ("quảng nam", 1), ("quảng ngãi", 1), ("bình định", 1),
    ("phú yên", 1), ("khánh hòa", 1), ("ninh thuận", 1), ("bình thuận", 1),
    ("kon tum", 1), ("gia lai", 1), ("đắk lắk", 1), ("đắk nông", 1),
    ("lâm đồng", 1), ("hà tĩnh", 1), ("nghệ an", 1), ("thanh hóa", 1),
    # Southern (2)
    ("hồ chí minh", 2), ("bình dương", 2), ("bình phước", 2),
    ("đồng nai", 2), ("tây ninh", 2), ("bà rịa", 2), ("long an", 2),
    ("tiền giang", 2), ("bến tre", 2), ("trà vinh", 2), ("vĩnh long", 2),
    ("đồng tháp", 2), ("an giang", 2), ("kiên giang", 2), ("cần thơ", 2),
    ("hậu giang", 2), ("sóc trăng", 2), ("bạc liêu", 2), ("cà mau", 2),
]


def province_to_dialect_idx(province: str) -> int:
    """Map a province name string to dialect index (0/1/2) or -1 if unknown."""
    lower = province.lower().strip()
    for key, idx in _PROVINCE_MAP:
        if key in lower:
            return idx
    return -1


class DialectClassifier(nn.Module):
    """
    Lightweight two-layer MLP that classifies dialect from encoder output.

    Architecture:
        encoder_hidden (B, T, D)
            → mean-pool over time → (B, D)
            → Linear(D, hidden_dim) → GELU → Dropout
            → Linear(hidden_dim, num_dialects)
            → logits (B, num_dialects)

    Parameters:  D * hidden_dim + hidden_dim * num_dialects  ≈  330K for D=1280
    This is tiny relative to the 1.55B parameter Whisper large-v3.
    """

    def __init__(self, d_model: int, num_dialects: int = 3, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.num_dialects = num_dialects
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dialects),
        )

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_hidden: (B, T, D) — encoder last hidden states
        Returns:
            logits: (B, num_dialects)
        """
        pooled = encoder_hidden.mean(dim=1)  # (B, D)
        return self.net(pooled)

    def compute_aux_loss(
        self,
        encoder_hidden: torch.Tensor,
        dialect_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute classification loss and accuracy, ignoring unknown samples (id=-1).

        Args:
            encoder_hidden: (B, T, D)
            dialect_ids:    (B,) — values in {0, 1, 2, -1}

        Returns:
            loss: scalar tensor (0.0 if no valid labels in batch)
            acc:  scalar tensor — fraction correctly classified
        """
        logits = self.forward(encoder_hidden)   # (B, num_dialects)
        valid = dialect_ids >= 0

        if not valid.any():
            zero = torch.tensor(0.0, device=encoder_hidden.device, requires_grad=True)
            return zero, zero.detach()

        loss = F.cross_entropy(logits[valid], dialect_ids[valid])
        acc = (logits[valid].argmax(dim=-1) == dialect_ids[valid]).float().mean()
        return loss, acc.detach()

    @torch.no_grad()
    def predict(self, encoder_hidden: torch.Tensor) -> list[str]:
        """Return dialect name predictions for a batch."""
        logits = self.forward(encoder_hidden)
        indices = logits.argmax(dim=-1).tolist()
        return [IDX_TO_DIALECT.get(i, "Unknown") for i in indices]
