"""
Whisper model builder: LoRA via PEFT + Flash Attention 2 + optional DAAT.

Load order (matters for correctness):
  1. Load base model with Flash Attention 2 and BF16 base weights
  2. Inject MoE layers (if enabled) — before PEFT so MoE params get LoRA'd
  3. Apply LoRA via PEFT
  4. Enable gradient checkpointing AFTER PEFT with use_reentrant=False
  5. Optional torch.compile (disabled by default — enable after stability check)
  6. Build dialect classifier as a separate module (not wrapped by PEFT)
"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.models.dialect_adapter import DialectClassifier
from src.models.moe import inject_moe_into_whisper


def build_whisper_lora_model(
    cfg: dict,
) -> tuple:
    """
    Build Whisper with LoRA (and optionally MoE + dialect classifier).

    Returns:
        model:              PEFT-wrapped WhisperForConditionalGeneration
        processor:          WhisperProcessor
        dialect_classifier: DialectClassifier | None
    """
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    moe_cfg = cfg.get("moe", {})
    daat_cfg = cfg.get("dialect_adapter", {})

    model_name = model_cfg["name"]
    use_flash = model_cfg.get("flash_attention", True)
    print(f"Loading Whisper model: {model_name}")
    print(f"  Flash Attention 2: {use_flash}")

    # ── 1. Load base model ────────────────────────────────────────────────────
    attn_impl = "flash_attention_2" if use_flash else "sdpa"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        # Load base weights in BF16 to halve the static memory footprint.
        # LoRA params are initialized in FP32 and cast back below.
        torch_dtype=torch.bfloat16,
    )

    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=model_cfg.get("language", "vi"),
        task=model_cfg.get("task", "transcribe"),
    )

    # Disable caching by default (required during training); generation re-enables it
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # ── 2. Freeze layers (optional) ───────────────────────────────────────────
    if model_cfg.get("freeze_encoder", False):
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        print("  Encoder frozen")

    if model_cfg.get("freeze_embed", False):
        _freeze_embeddings(model)
        print("  Embeddings frozen")

    # ── 3. Inject MoE before LoRA ─────────────────────────────────────────────
    if moe_cfg.get("enabled", False):
        model = inject_moe_into_whisper(model, moe_cfg)

    # ── 4. Apply LoRA ─────────────────────────────────────────────────────────
    if lora_cfg.get("enabled", True):
        model = _apply_lora(model, lora_cfg)

    # ── 5. Gradient checkpointing (AFTER PEFT, use_reentrant=False) ───────────
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  Gradient checkpointing enabled")

    # ── 6. torch.compile (optional) ───────────────────────────────────────────
    if model_cfg.get("torch_compile", False):
        model = torch.compile(model, dynamic=True, mode="reduce-overhead")
        print("  torch.compile applied (dynamic=True, mode=reduce-overhead)")

    _print_trainable_params(model)

    # ── 7. Dialect classifier (separate module, not wrapped by PEFT) ──────────
    dialect_classifier = None
    if daat_cfg.get("enabled", False):
        d_model = _get_encoder_dim(model)
        dialect_classifier = DialectClassifier(
            d_model=d_model,
            num_dialects=daat_cfg.get("num_dialects", 3),
            hidden_dim=daat_cfg.get("hidden_dim", 256),
            dropout=daat_cfg.get("dropout", 0.1),
        )
        print(f"  Dialect classifier: d_model={d_model}, "
              f"dialects={daat_cfg.get('num_dialects', 3)}, "
              f"hidden={daat_cfg.get('hidden_dim', 256)}")

    return model, processor, dialect_classifier


def _apply_lora(model, lora_cfg: dict):
    """Wrap model with PEFT LoRA and keep LoRA weights in FP32."""
    config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, config)

    # Cast only trainable (LoRA) params back to FP32 for training stability.
    # Base weights remain in BF16 to save ~3 GB for large-v3.
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    print(f"  LoRA applied: r={config.r}, alpha={config.lora_alpha}, "
          f"targets={config.target_modules}")
    return model


def _freeze_embeddings(model) -> None:
    if hasattr(model.model, "decoder"):
        for p in model.model.decoder.embed_tokens.parameters():
            p.requires_grad = False
        for p in model.model.decoder.embed_positions.parameters():
            p.requires_grad = False
    if hasattr(model.model, "encoder"):
        for p in model.model.encoder.conv1.parameters():
            p.requires_grad = False
        for p in model.model.encoder.conv2.parameters():
            p.requires_grad = False


def _get_encoder_dim(model) -> int:
    """Resolve encoder hidden dimension regardless of PEFT wrapping."""
    config = getattr(model, "config", None) or getattr(model, "base_model", model).config
    return getattr(config, "d_model", 1280)


def _print_trainable_params(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    print(f"  Parameters: {total:,} total | {trainable:,} trainable ({pct:.2f}%)")


def load_finetuned_model(base_model_name: str, adapter_path: str):
    """Load a saved LoRA adapter and merge it into the base model for inference."""
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    processor = WhisperProcessor.from_pretrained(base_model_name)
    return model, processor
