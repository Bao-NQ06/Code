"""
Main training script — Whisper Vietnamese fine-tuning on RTX 5090.

Usage:
    python train.py
    python train.py --model openai/whisper-large-v3 --batch_size 32
    python train.py --lora_r 128 --lr 3e-5
    python train.py --moe_enabled true
    python train.py --resume_from outputs/checkpoints/last.ckpt
"""

import argparse
import gc
import os

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

from src.data.datamodule import WhisperDataModule
from src.models.whisper_lora import build_whisper_lora_model
from src.models.lightning_module import WhisperFineTuneModule


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model:
        cfg["model"]["name"] = args.model
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    if args.data_dir:
        cfg["data"]["train_data_dir"] = args.data_dir
    if args.lora_r:
        cfg["lora"]["r"] = args.lora_r
    if args.moe_enabled is not None:
        cfg["moe"]["enabled"] = args.moe_enabled
    if args.freeze_encoder is not None:
        cfg["model"]["freeze_encoder"] = args.freeze_encoder
    if args.devices:
        cfg["hardware"]["devices"] = args.devices
    return cfg


# ── Callbacks & loggers ───────────────────────────────────────────────────────

def build_callbacks(cfg: dict) -> list:
    train_cfg = cfg["training"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]

    return [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="whisper-vi-{epoch:02d}-{val_wer:.4f}",
            monitor="val_wer",
            mode="min",
            save_top_k=train_cfg.get("save_top_k", 3),
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_wer",
            mode="min",
            patience=train_cfg.get("early_stopping_patience", 5),
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]


def build_logger(cfg: dict):
    log_cfg = cfg.get("logging", {})
    paths_cfg = cfg["paths"]
    wandb_cfg = log_cfg.get("wandb", {})
    loggers = []

    for lt in [s.strip() for s in log_cfg.get("logger", "csv").split(",")]:
        if lt == "wandb":
            loggers.append(_build_wandb_logger(cfg, log_cfg, wandb_cfg, paths_cfg))
        elif lt == "tensorboard":
            loggers.append(TensorBoardLogger(
                save_dir=paths_cfg["log_dir"],
                name=log_cfg.get("project_name", "whisper-vi"),
            ))
        elif lt == "csv":
            loggers.append(CSVLogger(
                save_dir=paths_cfg["log_dir"],
                name=log_cfg.get("project_name", "whisper-vi"),
            ))

    if not loggers:
        loggers.append(CSVLogger(save_dir=paths_cfg["log_dir"]))

    return loggers if len(loggers) > 1 else loggers[0]


def _build_wandb_logger(cfg, log_cfg, wandb_cfg, paths_cfg) -> WandbLogger:
    hparams = {
        "model": cfg["model"]["name"],
        "lora_r": cfg["lora"].get("r"),
        "lora_alpha": cfg["lora"].get("alpha"),
        "moe_enabled": cfg.get("moe", {}).get("enabled"),
        "daat_enabled": cfg.get("dialect_adapter", {}).get("enabled"),
        "daat_weight": cfg.get("dialect_adapter", {}).get("aux_loss_weight"),
        "batch_size": cfg["training"]["batch_size"],
        "lr": cfg["training"]["learning_rate"],
        "max_epochs": cfg["training"]["max_epochs"],
        "precision": cfg["hardware"].get("precision"),
        "flash_attention": cfg["model"].get("flash_attention"),
    }
    return WandbLogger(
        project=wandb_cfg.get("project", "whisper-vi-finetune"),
        name=wandb_cfg.get("run_name"),
        save_dir=paths_cfg["log_dir"],
        log_model=wandb_cfg.get("log_model", False),
        tags=wandb_cfg.get("tags", []),
        notes=wandb_cfg.get("notes", ""),
        config=hparams,
        group=wandb_cfg.get("group"),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Whisper Vietnamese Fine-tuning")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--moe_enabled", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--freeze_encoder", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args)
    pl.seed_everything(args.seed, workers=True)

    for d in cfg["paths"].values():
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("Building model…")
    print("=" * 60)
    model, processor, dialect_classifier = build_whisper_lora_model(cfg)
    datamodule = WhisperDataModule(cfg)
    module = WhisperFineTuneModule(model, processor, cfg, dialect_classifier)

    # Move dialect classifier to same device as model (Lightning handles the rest)
    if dialect_classifier is not None:
        module.dialect_classifier = dialect_classifier

    hw = cfg["hardware"]
    train_cfg = cfg["training"]

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        max_steps=train_cfg.get("max_steps", -1),
        accelerator=hw.get("accelerator", "gpu"),
        devices=hw.get("devices", 1),
        strategy=hw.get("strategy", "auto"),
        precision=hw.get("precision", "bf16-mixed"),
        accumulate_grad_batches=train_cfg.get("gradient_accumulation_steps", 1),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        val_check_interval=train_cfg.get("val_check_interval", 0.25),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        num_sanity_val_steps=1,   # RTX 5090: sanity check is safe
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg),
        deterministic=False,      # Non-deterministic for Flash Attn + compile speed
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
              f"{allocated:.1f}/{total:.1f} GB allocated  |  "
              f"{reserved:.1f} GB reserved")

    print("=" * 60)
    print("Starting training…")
    print("=" * 60)
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume_from)

    print("=" * 60)
    print("Running test evaluation…")
    print("=" * 60)
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
