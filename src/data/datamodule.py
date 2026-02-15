"""
PyTorch Lightning DataModule for Whisper Vietnamese fine-tuning.
"""

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from src.data.dataset import VietnameseASRDataset, DataCollatorSpeechSeq2Seq


class WhisperDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping VietnameseASRDataset."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        self.data_dir = data_cfg["train_data_dir"]
        self.val_split = data_cfg.get("val_split", 0.05)
        self.test_split = data_cfg.get("test_split", 0.05)
        self.seed = data_cfg.get("seed", 42)
        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = data_cfg.get("num_workers", 4)
        self.pin_memory = data_cfg.get("pin_memory", True)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_cfg["name"]
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_cfg["name"],
            language=model_cfg.get("language", "vi"),
            task=model_cfg.get("task", "transcribe"),
        )
        self.collator = DataCollatorSpeechSeq2Seq(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )

        # Optionally import normalize function
        self.normalize_fn = None
        try:
            from src.utils.text_normalize import normalize_vietnamese
            self.normalize_fn = normalize_vietnamese
        except ImportError:
            pass

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """Load data and split into train/val/test."""
        full_dataset = VietnameseASRDataset(
            data_dir=self.data_dir,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            max_audio_length=self.cfg["data"].get("max_audio_length", 30.0),
            sample_rate=self.cfg["data"].get("sample_rate", 16000),
            audio_column=self.cfg["data"].get("audio_column", "audio"),
            text_column=self.cfg["data"].get("text_column", "sentence"),
            normalize_fn=self.normalize_fn,
        )

        total = len(full_dataset)
        val_size = int(total * self.val_split)
        test_size = int(total * self.test_split)
        train_size = total - val_size - test_size

        import torch
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        print(
            f"Split: train={train_size}, val={val_size}, test={test_size}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
