"""
Vietnamese ASR Dataset - Ultra-low-memory loading from parquet files.

Reads parquet files at the ROW GROUP level (not full file), extracts
only the single row needed, and immediately frees memory. This keeps
peak RAM under control even on Colab's 12.7 GB limit.

Peak RAM from data loading: ~200-400 MB (one row group, not full file)
"""

import bisect
import gc
import glob
import io

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from transformers import WhisperFeatureExtractor, WhisperTokenizer


class VietnameseASRDataset(Dataset):
    """Ultra-low-memory dataset using row-group-level parquet reading.

    Builds a fine-grained index: (file, row_group, local_row) for each
    sample. Only one row group (~100MB) is ever in RAM at a time.
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
        self._columns = [audio_column, text_column]

        # Index: list of (file_path, row_group_idx, row_in_rg, rg_num_rows)
        self._index = []
        self._build_index(data_dir)

        # Cache one row group at a time
        self._cache_key = None  # (file_path, rg_idx)
        self._cache_data = None  # dict of column -> list

        print(f"Indexed {len(self._index)} samples (zero data loaded)")

    def _build_index(self, data_dir: str):
        """Scan parquet metadata at the row-group level."""
        candidates = sorted(glob.glob(f"{data_dir}/*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet files in {data_dir}")

        skipped = []
        total_rgs = 0

        for fpath in candidates:
            try:
                pf = pq.ParquetFile(fpath)
                meta = pf.metadata
                for rg_idx in range(meta.num_row_groups):
                    rg_rows = meta.row_group(rg_idx).num_rows
                    for row_in_rg in range(rg_rows):
                        self._index.append((fpath, rg_idx, row_in_rg))
                    total_rgs += 1
            except Exception as e:
                skipped.append((fpath, str(e)[:80]))

        if skipped:
            print(f"  WARNING: Skipped {len(skipped)} invalid file(s):")
            for p, r in skipped:
                print(f"    - {p}: {r}")

        if not self._index:
            raise FileNotFoundError(f"No valid data in {data_dir}")

        print(f"  Scanned: {len(candidates) - len(skipped)} files, "
              f"{total_rgs} row groups, {len(self._index)} rows")

    def __len__(self) -> int:
        return len(self._index)

    def _read_row_group(self, fpath: str, rg_idx: int) -> dict:
        """Read one row group and cache it. Free old cache first."""
        cache_key = (fpath, rg_idx)
        if self._cache_key == cache_key:
            return self._cache_data

        # Free old cache before loading new one
        self._cache_data = None
        self._cache_key = None
        gc.collect()

        pf = pq.ParquetFile(fpath)
        rg_table = pf.read_row_group(rg_idx, columns=self._columns)
        data = rg_table.to_pydict()
        del rg_table
        gc.collect()

        self._cache_data = data
        self._cache_key = cache_key
        return data

    def __getitem__(self, idx: int) -> dict:
        fpath, rg_idx, row_in_rg = self._index[idx]

        try:
            data = self._read_row_group(fpath, rg_idx)
            audio_entry = data[self.audio_column][row_in_rg]
            text = str(data[self.text_column][row_in_rg])
        except Exception as e:
            print(f"WARNING: Error reading sample {idx}: {e}")
            return self._fallback_sample()

        try:
            audio_array = self._extract_audio(audio_entry)
        except Exception as e:
            print(f"WARNING: Audio decode error sample {idx}: {e}")
            return self._fallback_sample()

        max_samples = int(self.max_audio_length * self.sample_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]

        if self.normalize_fn:
            text = self.normalize_fn(text)
        labels = self.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}

    def _fallback_sample(self) -> dict:
        """Return silent audio + empty text for failed samples."""
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        features = self.feature_extractor(
            silence, sampling_rate=self.sample_rate, return_tensors="pt",
        ).input_features[0]
        return {"input_features": features, "labels": self.tokenizer("").input_ids}

    def _extract_audio(self, audio_entry) -> np.ndarray:
        """Extract audio from various parquet storage formats."""
        if isinstance(audio_entry, dict):
            if "array" in audio_entry:
                return np.array(audio_entry["array"], dtype=np.float32)
            if "bytes" in audio_entry and audio_entry["bytes"]:
                data, _ = sf.read(io.BytesIO(audio_entry["bytes"]))
                return data.astype(np.float32)
            if "path" in audio_entry and audio_entry["path"]:
                data, _ = sf.read(audio_entry["path"])
                return data.astype(np.float32)
        if isinstance(audio_entry, (list, np.ndarray)):
            return np.array(audio_entry, dtype=np.float32)
        if isinstance(audio_entry, bytes):
            data, _ = sf.read(io.BytesIO(audio_entry))
            return data.astype(np.float32)
        raise ValueError(f"Unsupported audio format: {type(audio_entry)}")


class DataCollatorSpeechSeq2Seq:
    """Collator that pads input features and labels for Whisper."""

    def __init__(self, feature_extractor, tokenizer, pad_to_multiple_of=None):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict:
        input_features = torch.stack([f["input_features"] for f in features])

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_features, padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}
