"""
PyTorch Lightning module for Whisper fine-tuning with LoRA + optional MoE.
"""

import torch
import pytorch_lightning as pl
from transformers import WhisperProcessor, get_scheduler

from src.utils.metrics import compute_wer, compute_cer


class WhisperFineTuneModule(pl.LightningModule):
    """Lightning module wrapping Whisper with LoRA for training."""

    def __init__(self, model, processor: WhisperProcessor, cfg: dict):
        super().__init__()
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.moe_enabled = cfg.get("moe", {}).get("enabled", False)
        self.moe_loss_weight = cfg.get("moe", {}).get(
            "load_balance_weight", 0.01
        )
        self.save_hyperparameters(ignore=["model", "processor"])

    def forward(self, input_features, labels=None):
        return self.model(input_features=input_features, labels=labels)

    def _compute_loss(self, batch):
        """Compute loss including optional MoE auxiliary loss."""
        outputs = self.model(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Add MoE load-balancing auxiliary loss
        if self.moe_enabled:
            aux_loss = self._collect_moe_aux_loss()
            if aux_loss is not None:
                loss = loss + self.moe_loss_weight * aux_loss
                self.log("moe_aux_loss", aux_loss, prog_bar=False)

        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Decode predictions and compute metrics
        metrics = self._decode_and_eval(batch)
        self.log("val_wer", metrics["wer"], prog_bar=True, sync_dist=True)
        self.log("val_cer", metrics["cer"], prog_bar=True, sync_dist=True)
        return {"val_loss": loss, **metrics}

    def test_step(self, batch, batch_idx):
        loss, _ = self._compute_loss(batch)
        metrics = self._decode_and_eval(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_wer", metrics["wer"], sync_dist=True)
        self.log("test_cer", metrics["cer"], sync_dist=True)
        return {"test_loss": loss, **metrics}

    def _decode_and_eval(self, batch) -> dict:
        """Generate transcriptions and compute WER/CER."""
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features=batch["input_features"],
                language="vi",
                task="transcribe",
            )

        # Decode predictions and references
        preds = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        labels = batch["labels"].clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        refs = self.processor.batch_decode(
            labels, skip_special_tokens=True
        )

        wer = compute_wer(predictions=preds, references=refs)
        cer = compute_cer(predictions=preds, references=refs)
        return {"wer": wer, "cer": cer, "preds": preds, "refs": refs}

    def _collect_moe_aux_loss(self) -> torch.Tensor | None:
        """Collect MoE auxiliary losses from all MoE layers."""
        total_aux = None
        base_model = (
            self.model.base_model
            if hasattr(self.model, "base_model")
            else self.model
        )
        core = getattr(base_model, "model", base_model)

        for name in ["decoder", "encoder"]:
            module = getattr(core, name, None)
            if module is None:
                continue
            for layer in module.layers:
                aux = getattr(layer, "_moe_aux_loss", None)
                if aux is not None:
                    total_aux = aux if total_aux is None else total_aux + aux
                    layer._moe_aux_loss = None  # Reset

        return total_aux

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        train_cfg = self.cfg["training"]
        opt_cfg = self.cfg.get("optimizer", {})

        # Filter trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        # Compute total training steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = opt_cfg.get(
            "num_warmup_steps", train_cfg.get("warmup_steps", 500)
        )

        scheduler = get_scheduler(
            name=opt_cfg.get("scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
