"""
Vietnamese ASR Dataset - Loads preprocessed parquet files for Whisper fine-tuning.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer


class VietnameseASRDataset(Dataset):
    """Dataset for Vietnamese speech recognition from parquet files.

    Each parquet file contains columns:
        - audio: dict with 'array' (waveform) and 'sampling_rate'
        - sentence: transcript text
        - province_name, gender, speaker_id (metadata)
    """

    def __init__(
        self,
        data_dir: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        max_audio_length: float = 30.0,
        sample_rate: int = 16000,
        audio_column: str = "audio",
        text_column: str = "sentence",
        normalize_fn=None,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.audio_column = audio_column
        self.text_column = text_column
        self.normalize_fn = normalize_fn

        self.data = self._load_parquet_files(data_dir)
        print(f"Loaded {len(self.data)} samples from {data_dir}")

    def _load_parquet_files(self, data_dir: str) -> pd.DataFrame:
        """Load and concatenate all parquet files from the directory."""
        parquet_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        ])
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {data_dir}"
            )

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    def __len__(self) -> int:
        return len(self.data)

    def _extract_audio_array(self, audio_entry) -> np.ndarray:
        """Extract audio numpy array from the parquet audio column."""
        if isinstance(audio_entry, dict):
            return np.array(audio_entry["array"], dtype=np.float32)
        if isinstance(audio_entry, (list, np.ndarray)):
            return np.array(audio_entry, dtype=np.float32)
        if isinstance(audio_entry, bytes):
            return np.frombuffer(audio_entry, dtype=np.float32)
        raise ValueError(f"Unsupported audio format: {type(audio_entry)}")

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]

        # --- Audio processing ---
        audio_array = self._extract_audio_array(row[self.audio_column])
        max_samples = int(self.max_audio_length * self.sample_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]

        # --- Text processing ---
        text = str(row[self.text_column])
        if self.normalize_fn:
            text = self.normalize_fn(text)

        labels = self.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


class DataCollatorSpeechSeq2Seq:
    """Collator that pads input features and labels for Whisper."""

    def __init__(self, feature_extractor, tokenizer, pad_to_multiple_of=None):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict:
        input_features = [f["input_features"] for f in features]
        input_features = torch.stack(input_features)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Replace padding token id with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present (Whisper decoder adds it)
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }
