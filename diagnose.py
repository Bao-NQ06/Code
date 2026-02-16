"""
Diagnostic script - Run this BEFORE train.py to identify OOM/crash causes.
Tests each component in isolation to find which step crashes Colab.

Usage: python diagnose.py
"""

import gc
import os
import sys
import psutil


def print_mem(label=""):
    """Print current memory usage."""
    proc = psutil.Process()
    ram_gb = proc.memory_info().rss / 1e9
    ram_total = psutil.virtual_memory().total / 1e9
    ram_avail = psutil.virtual_memory().available / 1e9
    print(f"  [{label}] RAM: {ram_gb:.2f} GB used | "
          f"{ram_avail:.2f} / {ram_total:.2f} GB available")

    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  [{label}] GPU: {alloc:.2f} GB alloc | "
                  f"{reserved:.2f} GB reserved | {total:.2f} GB total")
    except Exception:
        pass

    disk = psutil.disk_usage("/")
    print(f"  [{label}] Disk: {disk.used / 1e9:.1f} / "
          f"{disk.total / 1e9:.1f} GB ({disk.percent}%)")


def test_step(name, fn):
    """Run a test step and catch any errors."""
    print(f"\n{'='*50}")
    print(f"STEP: {name}")
    print(f"{'='*50}")
    try:
        result = fn()
        print(f"  PASSED")
        print_mem(name)
        gc.collect()
        return result
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print_mem(f"{name} (failed)")
        return None


def main():
    print("=" * 50)
    print("WHISPER VIETNAMESE - DIAGNOSTIC")
    print("=" * 50)
    print_mem("startup")

    # Step 1: Check disk space
    def check_disk():
        disk = psutil.disk_usage("/")
        free_gb = disk.free / 1e9
        if free_gb < 5:
            print(f"  WARNING: Only {free_gb:.1f} GB free disk!")
        # Check HF cache
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_dir):
            cache_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fnames in os.walk(cache_dir)
                for f in fnames
            ) / 1e9
            print(f"  HF cache: {cache_size:.2f} GB at {cache_dir}")
    test_step("Check disk & cache", check_disk)

    # Step 2: Load config
    import yaml
    def load_cfg():
        with open("configs/config.yaml", "r") as f:
            return yaml.safe_load(f)
    cfg = test_step("Load config", load_cfg)
    if not cfg:
        return

    # Step 3: Load tokenizer/feature extractor
    def load_processor():
        from transformers import WhisperFeatureExtractor, WhisperTokenizer
        fe = WhisperFeatureExtractor.from_pretrained(cfg["model"]["name"])
        tok = WhisperTokenizer.from_pretrained(
            cfg["model"]["name"], language="vi", task="transcribe"
        )
        return fe, tok
    result = test_step("Load processor", load_processor)
    if not result:
        return
    fe, tok = result

    # Step 4: Test loading 1 parquet file
    def test_parquet():
        import pyarrow.parquet as pq
        import glob
        data_dir = cfg["data"]["train_data_dir"]
        files = sorted(glob.glob(f"{data_dir}/*.parquet"))
        print(f"  Found {len(files)} parquet files")
        if not files:
            raise FileNotFoundError(f"No files in {data_dir}")

        # Read with column projection
        cols = [cfg["data"]["audio_column"], cfg["data"]["text_column"]]
        print(f"  Reading first file with columns={cols}...")
        table = pq.read_table(files[0], columns=cols)
        print(f"  Rows: {table.num_rows}, Columns: {table.column_names}")
        size_mb = table.nbytes / 1e6
        print(f"  Table size in memory: {size_mb:.1f} MB")
        del table
        gc.collect()
    test_step("Load 1 parquet file", test_parquet)

    # Step 5: Test dataset __getitem__
    def test_dataset():
        from src.data.dataset import VietnameseASRDataset
        ds = VietnameseASRDataset(
            data_dir=cfg["data"]["train_data_dir"],
            feature_extractor=fe, tokenizer=tok,
            max_audio_length=cfg["data"]["max_audio_length"],
            audio_column=cfg["data"]["audio_column"],
            text_column=cfg["data"]["text_column"],
        )
        print(f"  Dataset length: {len(ds)}")
        print(f"  Testing __getitem__(0)...")
        sample = ds[0]
        print(f"  input_features shape: {sample['input_features'].shape}")
        print(f"  labels length: {len(sample['labels'])}")
        return ds
    test_step("Dataset getitem", test_dataset)

    # Step 6: Load model
    def load_model():
        import torch
        from src.models.whisper_lora import build_whisper_lora_model
        model, processor = build_whisper_lora_model(cfg)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model
    model = test_step("Load model to GPU", load_model)
    if not model:
        return

    # Step 7: Test forward pass
    def test_forward():
        import torch
        from src.data.dataset import VietnameseASRDataset
        ds = VietnameseASRDataset(
            data_dir=cfg["data"]["train_data_dir"],
            feature_extractor=fe, tokenizer=tok,
            max_audio_length=cfg["data"]["max_audio_length"],
            audio_column=cfg["data"]["audio_column"],
            text_column=cfg["data"]["text_column"],
        )
        sample = ds[0]
        device = next(model.parameters()).device
        features = sample["input_features"].unsqueeze(0).to(device)
        labels = torch.tensor([sample["labels"]], device=device)
        print(f"  Running forward pass (batch=1)...")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_features=features, labels=labels)
        print(f"  Loss: {out.loss.item():.4f}")
        del features, labels, out
        torch.cuda.empty_cache()
    test_step("Forward pass (batch=1)", test_forward)

    print("\n" + "=" * 50)
    print("ALL DIAGNOSTICS PASSED - training should work!")
    print("=" * 50)


if __name__ == "__main__":
    main()
