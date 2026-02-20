"""
PyTorch Lightning module for Whisper fine-tuning.

With 32 GB VRAM on RTX 5090:
  - Full-batch generation during validation (no more 1-sample-at-a-time)
  - Dialect-Adaptive Auxiliary Task (DAAT) integrated into the training loss
  - Dialect classification accuracy tracked as a metric
  - Sanity check is kept (num_sanity_val_steps=1) — no OOM risk anymore
"""

import torch
import pytorch_lightning as pl
from transformers import WhisperProcessor, get_scheduler

from src.utils.metrics import compute_wer, compute_cer
from src.models.dialect_adapter import DialectClassifier


class WhisperFineTuneModule(pl.LightningModule):
    """Lightning module wrapping Whisper LoRA with optional DAAT."""

    def __init__(self, model, processor: WhisperProcessor, cfg: dict,
                 dialect_classifier: DialectClassifier | None = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.dialect_classifier = dialect_classifier

        self.moe_enabled = cfg.get("moe", {}).get("enabled", False)
        self.moe_loss_weight = cfg.get("moe", {}).get("load_balance_weight", 0.01)

        daat_cfg = cfg.get("dialect_adapter", {})
        self.daat_enabled = daat_cfg.get("enabled", False) and dialect_classifier is not None
        self.daat_weight = daat_cfg.get("aux_loss_weight", 0.1)

        wandb_cfg = cfg.get("logging", {}).get("wandb", {})
        self._log_predictions = wandb_cfg.get("log_predictions", False)

        self.save_hyperparameters(ignore=["model", "processor", "dialect_classifier"])

    def forward(self, input_features, labels=None):
        return self.model(input_features=input_features, labels=labels)

    # ── Loss computation ───────────────────────────────────────────────────────

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, object]:
        outputs = self.model(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        if self.moe_enabled:
            aux = self._collect_moe_aux_loss()
            if aux is not None:
                loss = loss + self.moe_loss_weight * aux
                self.log("moe_aux_loss", aux, prog_bar=False)

        if self.daat_enabled and "dialect_ids" in batch:
            enc_hidden = outputs.encoder_last_hidden_state  # (B, T, D) — free
            dialect_loss, dialect_acc = self.dialect_classifier.compute_aux_loss(
                enc_hidden, batch["dialect_ids"]
            )
            loss = loss + self.daat_weight * dialect_loss
            self.log("dialect_clf_loss", dialect_loss, prog_bar=False)
            self.log("dialect_clf_acc", dialect_acc, prog_bar=True)

        return loss, outputs

    # ── Training / Validation / Test steps ────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _ = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        loss, _ = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Full-batch generation — 32 GB VRAM handles it without tricks
        if batch_idx == 0 and not self.trainer.sanity_checking:
            metrics = self._decode_and_eval(batch)
            self.log("val_wer", metrics["wer"], prog_bar=True, sync_dist=True)
            self.log("val_cer", metrics["cer"], prog_bar=True, sync_dist=True)
            if self._log_predictions:
                self._log_wandb_table(metrics["preds"], metrics["refs"], "val")

        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        loss, _ = self._compute_loss(batch)
        self.log("test_loss", loss, sync_dist=True)
        metrics = self._decode_and_eval(batch)
        self.log("test_wer", metrics["wer"], sync_dist=True)
        self.log("test_cer", metrics["cer"], sync_dist=True)
        return {"test_loss": loss, **metrics}

    # ── Generation & evaluation ───────────────────────────────────────────────

    @torch.no_grad()
    def _decode_and_eval(self, batch: dict) -> dict:
        """Full-batch beam-search generation and metric computation."""
        was_training = self.model.training
        self.model.eval()
        self.model.config.use_cache = True
        try:
            gen_ids = self.model.generate(
                input_features=batch["input_features"],
                language="vi",
                task="transcribe",
                num_beams=5,
                max_new_tokens=225,
            )
        finally:
            self.model.config.use_cache = False
            if was_training:
                self.model.train()

        preds = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        refs = self.processor.batch_decode(labels, skip_special_tokens=True)

        return {
            "wer": compute_wer(preds, refs),
            "cer": compute_cer(preds, refs),
            "preds": preds,
            "refs": refs,
        }

    # ── Optimizer & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        train_cfg = self.cfg["training"]
        opt_cfg = self.cfg.get("optimizer", {})

        # Collect all trainable params: LoRA weights + dialect classifier
        param_groups = [
            {"params": [p for p in self.model.parameters() if p.requires_grad],
             "lr": train_cfg["learning_rate"]},
        ]
        if self.dialect_classifier is not None:
            param_groups.append(
                {"params": self.dialect_classifier.parameters(),
                 "lr": train_cfg["learning_rate"]}
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.98])),
            eps=opt_cfg.get("eps", 1e-6),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        warmup = opt_cfg.get("num_warmup_steps", train_cfg.get("warmup_steps", 500))
        scheduler = get_scheduler(
            name=opt_cfg.get("scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collect_moe_aux_loss(self) -> torch.Tensor | None:
        total = None
        base = getattr(self.model, "base_model", self.model)
        core = getattr(base, "model", base)
        for part in ["encoder", "decoder"]:
            module = getattr(core, part, None)
            if module is None:
                continue
            for layer in module.layers:
                aux = getattr(layer, "_moe_aux_loss", None)
                if aux is not None:
                    total = aux if total is None else total + aux
                    layer._moe_aux_loss = None
        return total

    def _log_wandb_table(self, preds: list, refs: list, prefix: str) -> None:
        try:
            import wandb
        except ImportError:
            return
        loggers = self.logger if isinstance(self.logger, list) else [self.logger]
        wb = next((lg for lg in loggers if isinstance(lg, pl.loggers.WandbLogger)), None)
        if wb is None:
            return
        table = wandb.Table(columns=["reference", "prediction", "wer", "cer"])
        for i in range(min(10, len(preds))):
            table.add_data(
                refs[i], preds[i],
                round(compute_wer([preds[i]], [refs[i]]), 4),
                round(compute_cer([preds[i]], [refs[i]]), 4),
            )
        wb.experiment.log({f"{prefix}_predictions": table,
                           "global_step": self.global_step})
