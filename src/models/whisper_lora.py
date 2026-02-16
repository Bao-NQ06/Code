"""
Whisper model with LoRA (Low-Rank Adaptation) via PEFT.
Supports optional MoE (Mixture of Experts) integration.
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.models.moe import inject_moe_into_whisper


def build_whisper_lora_model(cfg: dict):
    """Build a Whisper model with LoRA and optional MoE.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        model: The Whisper model with LoRA (and optionally MoE).
        processor: WhisperProcessor for feature extraction and tokenization.
    """
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    moe_cfg = cfg.get("moe", {})

    model_name = model_cfg["name"]
    print(f"Loading Whisper model: {model_name}")

    # Load base model and processor
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=model_cfg.get("language", "vi"),
        task=model_cfg.get("task", "transcribe"),
    )

    # Configure generation settings (must use generation_config, not model.config)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False  # Required for gradient checkpointing

    # Freeze encoder if requested
    if model_cfg.get("freeze_encoder", False):
        print("Freezing encoder parameters")
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    # Freeze embedding layers if requested
    if model_cfg.get("freeze_embed", True):
        print("Freezing embedding parameters")
        _freeze_embeddings(model)

    # --- Apply MoE before LoRA if enabled ---
    if moe_cfg.get("enabled", False):
        print("Injecting MoE layers into Whisper")
        model = inject_moe_into_whisper(model, moe_cfg)

    # --- Apply LoRA FIRST, then gradient checkpointing ---
    if lora_cfg.get("enabled", True):
        model = _apply_lora(model, lora_cfg)

    # Gradient checkpointing MUST be enabled AFTER PEFT wrapping
    # and MUST use use_reentrant=False to work with LoRA
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled (use_reentrant=False)")

    _print_trainable_params(model)
    return model, processor


def _freeze_embeddings(model):
    """Freeze decoder embedding and encoder conv layers."""
    if hasattr(model.model, "decoder"):
        for param in model.model.decoder.embed_tokens.parameters():
            param.requires_grad = False
        for param in model.model.decoder.embed_positions.parameters():
            param.requires_grad = False

    if hasattr(model.model, "encoder"):
        for param in model.model.encoder.conv1.parameters():
            param.requires_grad = False
        for param in model.model.encoder.conv2.parameters():
            param.requires_grad = False


def _apply_lora(model, lora_cfg: dict):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    return model


def _print_trainable_params(model):
    """Print the number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total > 0 else 0
    print(
        f"Parameters: {total:,} total, {trainable:,} trainable ({pct:.2f}%)"
    )


def load_finetuned_model(base_model_name: str, adapter_path: str):
    """Load a fine-tuned LoRA model from checkpoint.

    Args:
        base_model_name: HuggingFace model name (e.g. 'openai/whisper-small').
        adapter_path: Path to saved LoRA adapter weights.

    Returns:
        model: Loaded model with LoRA adapters merged.
        processor: WhisperProcessor.
    """
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    processor = WhisperProcessor.from_pretrained(base_model_name)
    return model, processor
