"""
PyTorch Lightning DataModule for Whisper Vietnamese fine-tuning.

Split strategy: index-only random_split — no data is duplicated on disk or RAM.
Workers: 8 parallel workers with per-worker LRU file caches for fast I/O.
"""

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from src.data.dataset import VietnameseASRDataset, DataCollatorSpeechSeq2Seq


class WhisperDataModule(pl.LightningDataModule):
    """
    DataModule wrapping VietnameseASRDataset.

    The train/val/test split stores only integer index lists (random_split),
    so no audio data is ever copied. Each DataLoader worker independently
    maintains its own LRU parquet file cache.
    """

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
        self.num_workers = data_cfg.get("num_workers", 8)
        self.pin_memory = data_cfg.get("pin_memory", True)
        self.prefetch_factor = data_cfg.get("prefetch_factor", 4)
        self.cache_files = data_cfg.get("cache_files", 8)

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

        self.normalize_fn = None
        try:
            from src.utils.text_normalize import normalize_vietnamese
            self.normalize_fn = normalize_vietnamese
        except ImportError:
            pass

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        """Build index and create index-only train/val/test splits."""
        full_dataset = VietnameseASRDataset(
            data_dir=self.data_dir,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            max_audio_length=self.cfg["data"].get("max_audio_length", 30.0),
            sample_rate=self.cfg["data"].get("sample_rate", 16000),
            audio_column=self.cfg["data"].get("audio_column", "audio"),
            text_column=self.cfg["data"].get("text_column", "text"),
            province_column=self.cfg["data"].get("province_column", "province_name"),
            cache_files=self.cache_files,
            normalize_fn=self.normalize_fn,
        )

        total = len(full_dataset)
        val_n = int(total * self.val_split)
        test_n = int(total * self.test_split)
        train_n = total - val_n - test_n

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_n, val_n, test_n], generator=generator
        )
        print(f"  Split: train={train_n:,}  val={val_n:,}  test={test_n:,}")

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        # prefetch_factor only valid when num_workers > 0
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = True
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=shuffle,   # Drop last incomplete batch only during training
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)
