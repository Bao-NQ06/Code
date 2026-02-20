"""
Vietnamese ASR Dataset — Parquet loading with per-worker LRU file cache.

Strategy:
  - Index phase: read only parquet *metadata* (row counts). Zero data loaded.
  - Fetch phase: maintain a per-worker LRU cache of `cache_files` full parquet
    files in RAM. With 8 workers × cache_files=8, hot files stay resident and
    amortize disk I/O across epochs.
  - Column projection: only the audio and text columns are read from disk.

With 32+ GB system RAM this gives effectively unlimited caching, yet remains
correct when RAM is constrained (LRU evicts least-recently-used files).
"""

from __future__ import annotations

import gc
import glob
import io
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from src.models.dialect_adapter import province_to_dialect_idx


class _LRUFileCache:
    """Thread-unsafe LRU cache for parquet files (safe per DataLoader worker)."""

    def __init__(self, max_files: int):
        self._max = max_files
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, fpath: str, columns: list[str]) -> dict:
        if fpath in self._cache:
            self._cache.move_to_end(fpath)
            return self._cache[fpath]

        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
            gc.collect()

        table = pq.read_table(fpath, columns=columns)
        data = table.to_pydict()
        del table
        self._cache[fpath] = data
        self._cache.move_to_end(fpath)
        return data


class VietnameseASRDataset(Dataset):
    """
    Row-indexed dataset over a directory of parquet files.

    Each sample yields:
        input_features: (80, 3000) log-mel spectrogram
        labels:         list[int] — tokenized transcription
        dialect_id:     int — 0=Northern, 1=Central, 2=Southern, -1=Unknown
    """

    def __init__(
        self,
        data_dir: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        max_audio_length: float = 30.0,
        sample_rate: int = 16000,
        audio_column: str = "audio",
        text_column: str = "text",
        province_column: str = "province_name",
        cache_files: int = 8,
        normalize_fn=None,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.audio_column = audio_column
        self.text_column = text_column
        self.province_column = province_column
        self.normalize_fn = normalize_fn

        read_cols = [audio_column, text_column]
        if province_column:
            read_cols.append(province_column)
        self._columns = read_cols

        self._cache = _LRUFileCache(max_files=cache_files)

        # Index: list of (file_path, local_row_index)
        self._index: list[tuple[str, int]] = []
        self._build_index(data_dir)

        print(f"  Indexed {len(self._index)} samples across "
              f"{len(set(f for f, _ in self._index))} files (zero audio loaded)")

    def _build_index(self, data_dir: str) -> None:
        candidates = sorted(glob.glob(f"{data_dir}/*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet files found in: {data_dir}")

        skipped = []
        for fpath in candidates:
            try:
                meta = pq.read_metadata(fpath)
                n_rows = sum(
                    meta.row_group(i).num_rows
                    for i in range(meta.num_row_groups)
                )
                for row in range(n_rows):
                    self._index.append((fpath, row))
            except Exception as exc:
                skipped.append(f"{fpath}: {str(exc)[:80]}")

        if skipped:
            print(f"  WARNING: Skipped {len(skipped)} invalid file(s):")
            for msg in skipped:
                print(f"    - {msg}")
        if not self._index:
            raise FileNotFoundError(f"No valid parquet rows in: {data_dir}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fpath, row = self._index[idx]
        try:
            data = self._cache.get(fpath, self._columns)
            audio_entry = data[self.audio_column][row]
            text = str(data[self.text_column][row])
            province = str(data.get(self.province_column, [""])[row])
        except Exception as exc:
            print(f"WARNING: read error sample {idx} ({fpath}:{row}): {exc}")
            return self._fallback_sample()

        try:
            audio = self._extract_audio(audio_entry)
        except Exception as exc:
            print(f"WARNING: audio decode error sample {idx}: {exc}")
            return self._fallback_sample()

        max_samples = int(self.max_audio_length * self.sample_rate)
        audio = audio[:max_samples]

        features = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features[0]

        if self.normalize_fn:
            text = self.normalize_fn(text)
        labels = self.tokenizer(text).input_ids

        return {
            "input_features": features,
            "labels": labels,
            "dialect_id": province_to_dialect_idx(province),
        }

    def _fallback_sample(self) -> dict:
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        features = self.feature_extractor(
            silence, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features[0]
        return {
            "input_features": features,
            "labels": self.tokenizer("").input_ids,
            "dialect_id": -1,
        }

    @staticmethod
    def _extract_audio(entry) -> np.ndarray:
        """Decode audio from any parquet storage format."""
        if isinstance(entry, dict):
            if "array" in entry:
                return np.asarray(entry["array"], dtype=np.float32)
            if "bytes" in entry and entry["bytes"]:
                wav, _ = sf.read(io.BytesIO(entry["bytes"]))
                return wav.astype(np.float32)
            if "path" in entry and entry["path"]:
                wav, _ = sf.read(entry["path"])
                return wav.astype(np.float32)
        if isinstance(entry, (list, np.ndarray)):
            return np.asarray(entry, dtype=np.float32)
        if isinstance(entry, bytes):
            wav, _ = sf.read(io.BytesIO(entry))
            return wav.astype(np.float32)
        raise ValueError(f"Unsupported audio entry type: {type(entry)}")


class DataCollatorSpeechSeq2Seq:
    """Pad and batch input features, labels, and dialect IDs."""

    def __init__(self, feature_extractor: WhisperFeatureExtractor,
                 tokenizer: WhisperTokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        input_features = torch.stack([f["input_features"] for f in features])

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        dialect_ids = torch.tensor(
            [f.get("dialect_id", -1) for f in features], dtype=torch.long
        )

        return {
            "input_features": input_features,
            "labels": labels,
            "dialect_ids": dialect_ids,
        }
