"""
Evaluation script for fine-tuned Whisper Vietnamese model.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/last.ckpt
    python evaluate.py --adapter_path outputs/lora_adapter --base_model openai/whisper-small
    python evaluate.py --checkpoint outputs/checkpoints/last.ckpt --dialect_analysis
"""

import argparse
import json
import os

import yaml
import torch
import pandas as pd
from transformers import WhisperProcessor

from src.data.datamodule import WhisperDataModule
from src.models.whisper_lora import build_whisper_lora_model, load_finetuned_model
from src.models.lightning_module import WhisperFineTuneModule
from src.utils.metrics import compute_wer, compute_cer, dialect_wise_metrics
from src.utils.text_normalize import normalize_vietnamese


def evaluate_checkpoint(args):
    """Evaluate from a Lightning checkpoint."""
    cfg = load_config(args.config)

    model, processor = build_whisper_lora_model(cfg)
    module = WhisperFineTuneModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        processor=processor,
        cfg=cfg,
    )
    module.eval()
    return module, processor, cfg


def evaluate_adapter(args):
    """Evaluate from a saved LoRA adapter."""
    cfg = load_config(args.config)
    model, processor = load_finetuned_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
    )
    module = WhisperFineTuneModule(model, processor, cfg)
    module.eval()
    return module, processor, cfg


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_evaluation(module, processor, cfg, args):
    """Run evaluation on test set and print results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    # Build data module and get test loader
    datamodule = WhisperDataModule(cfg)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_refs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_features = batch["input_features"].to(device)
            generated_ids = module.model.generate(
                input_features=input_features,
                language="vi",
                task="transcribe",
            )
            preds = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Compute overall metrics
    overall_wer = compute_wer(all_preds, all_refs)
    overall_cer = compute_cer(all_preds, all_refs)

    results = {
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "num_samples": len(all_preds),
    }

    print("\n" + "=" * 50)
    print(f"Overall WER: {overall_wer:.4f} ({overall_wer * 100:.2f}%)")
    print(f"Overall CER: {overall_cer:.4f} ({overall_cer * 100:.2f}%)")
    print(f"Samples evaluated: {len(all_preds)}")
    print("=" * 50)

    # Dialect-wise analysis
    if args.dialect_analysis:
        results["dialect"] = _run_dialect_analysis(
            all_preds, all_refs, datamodule, cfg
        )

    # Save results
    output_path = args.output or "outputs/eval_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print sample predictions
    _print_samples(all_preds, all_refs, n=5)


def _run_dialect_analysis(preds, refs, datamodule, cfg):
    """Run dialect-wise WER/CER analysis."""
    province_col = cfg["data"].get("province_column", "province_name")
    full_data = datamodule.test_dataset.dataset.data

    test_indices = datamodule.test_dataset.indices
    provinces = [str(full_data.iloc[i][province_col]) for i in test_indices]

    dialect_results = dialect_wise_metrics(preds, refs, provinces)

    print("\nDialect-wise Results:")
    print("-" * 50)
    for region, metrics in sorted(dialect_results.items()):
        print(
            f"  {region:12s} | WER: {metrics['wer']:.4f} | "
            f"CER: {metrics['cer']:.4f} | N={metrics['count']}"
        )
    return dialect_results


def _print_samples(preds, refs, n=5):
    """Print sample predictions vs references."""
    print(f"\nSample Predictions (first {n}):")
    print("-" * 50)
    for i in range(min(n, len(preds))):
        print(f"  REF: {refs[i]}")
        print(f"  HYP: {preds[i]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper Vietnamese")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--dialect_analysis", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint:
        module, processor, cfg = evaluate_checkpoint(args)
    elif args.adapter_path:
        module, processor, cfg = evaluate_adapter(args)
    else:
        raise ValueError("Provide --checkpoint or --adapter_path")

    run_evaluation(module, processor, cfg, args)


if __name__ == "__main__":
    main()
