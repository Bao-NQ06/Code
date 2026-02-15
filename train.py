"""
Main training script for Whisper Vietnamese fine-tuning.

Usage:
    python train.py                          # Use default config
    python train.py --config configs/config.yaml
    python train.py --model openai/whisper-medium --batch_size 4
    python train.py --moe_enabled true       # Enable MoE
"""

import argparse
import os

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.data.datamodule import WhisperDataModule
from src.models.whisper_lora import build_whisper_lora_model
from src.models.lightning_module import WhisperFineTuneModule


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments."""
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


def build_callbacks(cfg: dict) -> list:
    """Build Lightning callbacks."""
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    callbacks = [
        ModelCheckpoint(
            dirpath=paths_cfg["checkpoint_dir"],
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
    return callbacks


def build_logger(cfg: dict):
    """Build Lightning logger."""
    log_cfg = cfg.get("logging", {})
    paths_cfg = cfg["paths"]
    logger_type = log_cfg.get("logger", "tensorboard")

    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=paths_cfg["log_dir"],
            name=log_cfg.get("project_name", "whisper-vi"),
        )
    return CSVLogger(
        save_dir=paths_cfg["log_dir"],
        name=log_cfg.get("project_name", "whisper-vi"),
    )


def main():
    parser = argparse.ArgumentParser(description="Whisper Vietnamese Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--moe_enabled", type=bool, default=None)
    parser.add_argument("--freeze_encoder", type=bool, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load and override config
    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    # Seed everything
    pl.seed_everything(args.seed, workers=True)

    # Create output directories
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    # Build components
    print("=" * 60)
    print("Building model and data module...")
    print("=" * 60)

    model, processor = build_whisper_lora_model(cfg)
    datamodule = WhisperDataModule(cfg)
    lightning_module = WhisperFineTuneModule(model, processor, cfg)

    # Build trainer
    hw_cfg = cfg["hardware"]
    train_cfg = cfg["training"]

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        max_steps=train_cfg.get("max_steps", -1),
        accelerator=hw_cfg.get("accelerator", "gpu"),
        devices=hw_cfg.get("devices", 1),
        strategy=hw_cfg.get("strategy", "auto"),
        precision=hw_cfg.get("precision", "16-mixed"),
        accumulate_grad_batches=train_cfg.get("gradient_accumulation_steps", 1),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        val_check_interval=train_cfg.get("val_check_interval", 0.25),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg),
        deterministic=False,
    )

    # Train
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=args.resume_from,
    )

    # Test
    print("=" * 60)
    print("Running test evaluation...")
    print("=" * 60)
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
