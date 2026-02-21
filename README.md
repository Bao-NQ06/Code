# Optimizing Whisper for Vietnamese Speech Recognition

> **DAT301 Capstone Project** — FPT University  
> Fine-tuning OpenAI Whisper using LoRA + Dialect-Adaptive Auxiliary Task (DAAT) for Vietnamese ASR  
> Hardware target: NVIDIA RTX 5090 (32 GB VRAM, 108.1 TFLOPS)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Contribution](#2-research-contribution)
3. [Repository Structure](#3-repository-structure)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [Architecture Deep-Dive](#5-architecture-deep-dive)
6. [Dataset](#6-dataset)
7. [Setup & Installation](#7-setup--installation)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Configuration Reference](#10-configuration-reference)
11. [Technical Decisions Log](#11-technical-decisions-log)
12. [Key Dependencies](#12-key-dependencies)

---

## 1. Project Overview

**Goal:** Develop a high-accuracy, low Word Error Rate (WER) Speech-to-Text (STT) system specifically for the Vietnamese language, covering all 63 former provincial dialects.

**Approach:**
- Fine-tune `openai/whisper-large-v3` (1.55 B parameters) using **LoRA** (Low-Rank Adaptation), a Parameter-Efficient Fine-Tuning (PEFT) method. Only ~2–3% of parameters are updated.
- Add a novel **Dialect-Adaptive Auxiliary Task (DAAT)** that jointly trains a dialect classifier alongside ASR, improving per-region accuracy through multi-task learning.
- Optionally inject **Mixture of Experts (MoE)** layers into the decoder FFN blocks for increased capacity.

**Why not full training?** Training Whisper from scratch requires thousands of A100 GPUs for months. Fine-tuning with LoRA on a single RTX 5090 is practical, achieves strong results, and is the academically accepted approach.

**Metrics evaluated:**
- Word Error Rate (WER) — primary metric
- Character Error Rate (CER) — captures tonal/phonetic errors specific to Vietnamese
- Dialect-wise WER/CER breakdown: Northern, Central, Southern regions

---

## 2. Research Contribution

### Dialect-Adaptive Auxiliary Task (DAAT)

**Problem:** Vietnamese ASR errors are not uniform across regions. Central dialect speakers (e.g., Huế, Đà Nẵng) suffer disproportionately high WER because a standard ASR encoder never learns explicit dialect-discriminative representations.

**Solution:** A lightweight 2-layer MLP classifier head is attached to the Whisper encoder's mean-pooled output. During fine-tuning, the total loss is:

```
L_total = L_ASR  +  λ · L_dialect_CE
```

where `λ = 0.1` (configurable), `L_ASR` is the standard cross-entropy over tokens, and `L_dialect_CE` is cross-entropy over 3 dialect classes (Northern=0, Central=1, Southern=2).

**Why this works:** The dialect gradient flows back through the LoRA-adapted encoder layers, forcing them to encode dialect identity alongside phonetic content. The encoder becomes more dialect-discriminative without any extra inference cost — the classifier head is discarded at inference time.

**Parameters:** ~330K extra params (2-layer MLP, `d_model=1280 → 256 → 3`). Negligible vs. the 1.55B base model.

**Paper angle:** *"Dialect-Adaptive Multi-Task LoRA Fine-tuning of Whisper for Low-Resource Vietnamese Speech Recognition"*

**Ablations to run:**
- Single LoRA vs. LoRA + DAAT (effect on per-region WER)
- λ sensitivity: 0.01, 0.05, 0.1, 0.5
- LoRA rank: r=16, 32, 64, 128
- Encoder frozen vs. unfrozen during LoRA fine-tuning
- With vs. without Flash Attention 2

---

## 3. Repository Structure

```
Code/
├── configs/
│   └── config.yaml                  # Central config for all hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # Parquet dataset with LRU file cache
│   │   └── datamodule.py            # PyTorch Lightning DataModule
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── whisper_lora.py          # Model builder: Flash Attn2 + LoRA + DAAT
│   │   ├── dialect_adapter.py       # DAAT: DialectClassifier + province mapping
│   │   ├── moe.py                   # Mixture of Experts (optional)
│   │   └── lightning_module.py      # Lightning training/val/test logic
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py               # WER, CER, dialect-wise metrics
│   │   └── text_normalize.py        # Vietnamese text normalization
│   │
│   └── demo/
│       ├── __init__.py
│       └── app.py                   # Gradio web demo for inference
│
├── train.py                         # Entry point for training
├── evaluate.py                      # Entry point for evaluation
├── diagnose.py                      # Debug: per-component memory diagnostics
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 4. File-by-File Reference

### `configs/config.yaml`
Central YAML configuration. Every hyperparameter lives here — no hardcoded values in source. CLI flags can override any top-level key at runtime.

**Sections:**
- `model`: Whisper model name, Flash Attention, torch.compile, gradient checkpointing
- `lora`: rank `r`, `alpha`, dropout, target modules, bias
- `dialect_adapter`: DAAT enabled flag, hidden dim, `aux_loss_weight` (λ)
- `moe`: enabled flag, expert count, top-k, capacity factor
- `data`: parquet directory, split ratios, num_workers, cache settings
- `training`: batch size, accumulation steps, learning rate, precision
- `optimizer`: AdamW betas/eps, LR scheduler type
- `logging`: W&B project name, tags, prediction logging
- `hardware`: accelerator, device count, precision string

---

### `src/data/dataset.py`
**Class `VietnameseASRDataset`** — `torch.utils.data.Dataset`

Index phase (startup):
- Scans all `.parquet` files with `pq.read_metadata()` — reads only row counts, zero audio loaded.
- Builds a flat index: `list[tuple[file_path, local_row_index]]`
- Invalid/corrupted files are skipped with a warning; training continues on valid files.

Fetch phase (`__getitem__`):
- Uses `_LRUFileCache` — an `OrderedDict`-based cache of `cache_files` (default 8) full parquet files per worker. Files are loaded with column projection (only audio + text + province columns), drastically reducing I/O.
- Decodes audio from multiple storage formats: `dict{"array": ...}`, `dict{"bytes": ...}`, raw `bytes`, or plain `list/ndarray`.
- Returns: `{"input_features": Tensor(80, 3000), "labels": list[int], "dialect_id": int}`

**Class `DataCollatorSpeechSeq2Seq`**
- Stacks pre-computed log-mel spectrograms (no re-extraction at batch time).
- Pads tokenized labels; masks padding with `-100` (ignored by cross-entropy).
- Removes BOS token if present (Whisper training convention).
- Collates `dialect_ids` as a `torch.long` tensor with `-1` for unknown provinces.

---

### `src/data/datamodule.py`
**Class `WhisperDataModule`** — `pl.LightningDataModule`

- `setup()`: creates one `VietnameseASRDataset` and splits it with `random_split` (index-only, no data duplication).
- `train_dataloader()`: `shuffle=True`, `drop_last=True`, `persistent_workers=True`
- `val/test_dataloader()`: `shuffle=False`, `drop_last=False`
- `prefetch_factor=4`: each of the 8 workers pre-loads 4 batches ahead, keeping GPU fed.

---

### `src/models/dialect_adapter.py`
**Novel research module.**

`province_to_dialect_idx(province: str) -> int`
- Substring-matches province names against a hand-built look-up table of all 63 provinces.
- Returns: 0 (Northern), 1 (Central), 2 (Southern), or -1 (Unknown).

`DialectClassifier(nn.Module)`
- `forward(encoder_hidden: Tensor[B, T, D]) -> Tensor[B, num_dialects]`
  - Mean-pools time dimension: `(B, T, D) → (B, D)`
  - MLP: `Linear(D, 256) → GELU → Dropout → Linear(256, 3)`
- `compute_aux_loss(encoder_hidden, dialect_ids) -> (loss, accuracy)`
  - Cross-entropy on valid samples only (skips `dialect_id == -1`).
  - Returns zero tensors if entire batch has unknown dialects.
- `predict(encoder_hidden) -> list[str]`
  - Returns dialect name strings (for inference/demo use).

---

### `src/models/whisper_lora.py`
**`build_whisper_lora_model(cfg) -> (model, processor, dialect_classifier)`**

Build order (order matters for correctness):
1. Load `WhisperForConditionalGeneration` with `attn_implementation="flash_attention_2"` and `torch_dtype=torch.bfloat16` (base weights in BF16 → saves ~3 GB VRAM for large-v3).
2. Freeze encoder or embeddings if configured.
3. Inject MoE layers (if `moe.enabled`).
4. Apply LoRA via `peft.get_peft_model()`. Then **cast LoRA params back to FP32** for training stability (base stays BF16, adapters stay FP32).
5. Enable gradient checkpointing **after** PEFT wrapping, with `use_reentrant=False` (required for PEFT compatibility).
6. Optional `torch.compile(model, dynamic=True, mode="reduce-overhead")`.
7. Build `DialectClassifier` as a **separate module** (not wrapped by PEFT) and return it alongside the model.

**`load_finetuned_model(base_model_name, adapter_path)`**
- Loads base model + LoRA adapter, calls `merge_and_unload()` to fuse weights for inference. No PEFT overhead at serving time.

---

### `src/models/lightning_module.py`
**Class `WhisperFineTuneModule`** — `pl.LightningModule`

`_compute_loss(batch)`:
1. Forward pass: `self.model(input_features, labels)` — returns `Seq2SeqLMOutput`
2. MoE aux loss: collected from `layer._moe_aux_loss` attributes set during forward.
3. DAAT aux loss: uses `outputs.encoder_last_hidden_state` (already computed, no extra encoder pass) and `self.dialect_classifier.compute_aux_loss()`.
4. Final: `L = L_ASR + λ_moe * L_moe + λ_daat * L_daat`

`_decode_and_eval(batch)`:
- Full-batch beam-search with `num_beams=5` (possible because 32 GB VRAM).
- Temporarily enables `use_cache=True` during generation (KV-cache speeds up autoregressive decoding), resets to `False` after.
- Returns WER, CER, predictions, references.

`configure_optimizers()`:
- AdamW with `β2=0.98, ε=1e-6` (Whisper paper values).
- Two parameter groups: LoRA params at `learning_rate`, dialect classifier at `learning_rate`.
- Cosine LR schedule with linear warmup.

---

### `src/models/moe.py`
**Classes:** `ExpertFFN`, `TopKRouter`, `MoELayer`

`TopKRouter`:
- Linear gate `(D → num_experts)` → softmax → top-k selection.
- Renormalizes selected weights to sum to 1.

`MoELayer.forward(x: Tensor[B, T, D])`:
- Flattens to `(N=B*T, D)` for vectorized dispatch.
- For each of the `top_k` slots: scatter tokens to the assigned expert, run expert FFN, scatter-add weighted output back.
- One kernel call per expert per slot (not per token) — GPU-efficient.
- Returns `(output, aux_loss)` where aux_loss is MSE between average routing probs and uniform distribution.

`inject_moe_into_whisper(model, moe_cfg)`:
- Monkey-patches `layer.forward` on every N-th decoder (or encoder) layer.
- Stores `aux_loss` on the layer object so Lightning's `_collect_moe_aux_loss()` can retrieve it.

---

### `src/utils/metrics.py`
- `compute_wer(preds, refs, normalize=True)` — calls `jiwer.wer()` after Vietnamese text normalization; filters empty references.
- `compute_cer(preds, refs, normalize=True)` — same with `jiwer.cer()`.
- `dialect_wise_metrics(preds, refs, provinces)` — groups by dialect region via `_province_to_region()`, returns per-region WER/CER/count dict.
- `DIALECT_REGIONS` — mapping from region name to list of province name strings.

---

### `src/utils/text_normalize.py`
Vietnamese-specific text normalization pipeline applied before WER/CER computation and optionally during training:
- Unicode NFC normalization
- Lowercase
- Punctuation removal
- Abbreviation expansion
- Whitespace normalization

---

### `train.py`
Entry point. Parses CLI args, loads + overrides config, builds all components, creates `pl.Trainer`, runs `trainer.fit()` then `trainer.test()`.

Key trainer settings:
- `precision="bf16-mixed"` — Lightning handles BF16 casting automatically.
- `num_sanity_val_steps=1` — safe to enable since 32 GB VRAM.
- `deterministic=False` — required for Flash Attention 2 and torch.compile.

Callbacks: `ModelCheckpoint` (monitors `val_wer`, saves top-3), `EarlyStopping` (patience=5), `LearningRateMonitor`, `RichProgressBar`.

---

### `evaluate.py`
Standalone evaluation script. Supports loading from a Lightning `.ckpt` file or a raw LoRA adapter directory. Runs inference on the test split, reports overall WER/CER, and optionally prints dialect-wise breakdown.

---

### `diagnose.py`
Step-by-step memory/disk diagnostics. Runs each pipeline component (config load → processor → parquet scan → dataset → model → forward pass) in isolation, printing RAM and GPU usage after each step. Use this when debugging crashes.

---

## 5. Architecture Deep-Dive

### Whisper Architecture (background)
Whisper is a Transformer encoder-decoder:
- **Encoder:** 2 Conv layers (audio → feature patches) → 32 Transformer layers (for large-v3) — processes the 80-channel log-mel spectrogram of shape `(80, 3000)` (30 seconds at 16 kHz).
- **Decoder:** 32 Transformer layers with cross-attention to encoder output — autoregressively generates token IDs.
- **Vocabulary:** ~51,865 multilingual tokens.

### LoRA (Low-Rank Adaptation)
Standard fine-tuning updates all weights: `W' = W + ΔW`. LoRA instead decomposes the update:

```
ΔW = B · A    where A ∈ R^{r×d}, B ∈ R^{d×r}, rank r << d
```

During training, `W` is frozen (BF16, no gradient) and only `A`, `B` are updated (FP32). At inference, weights can be merged: `W' = W + (α/r) · B · A`.

For large-v3 with `r=64`, `α=128`: ~26M trainable params out of 1,550M total (**1.7%**).

Applied to: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` in all attention and FFN blocks.

### Flash Attention 2
Rewrites the attention kernel to avoid materializing the full `(T, T)` attention matrix in HBM. Instead tiles the computation in SRAM. Effect: O(T) memory instead of O(T²), and ~2–3× faster for long sequences. Requires CUDA and `flash-attn` package. Falls back to PyTorch SDPA if unavailable.

### DAAT (Dialect-Adaptive Auxiliary Task)
```
Encoder hidden states (B, T, 1280)
    ↓ mean pool over T
(B, 1280)
    ↓ Linear(1280, 256) → GELU → Dropout(0.1)
(B, 256)
    ↓ Linear(256, 3)
(B, 3) = dialect logits
    ↓ cross-entropy with ground-truth dialect label
dialect_loss (scalar)
```
Ground truth dialect labels come from the `province_name` column in parquet → mapped to {0,1,2} via substring matching against a province registry. Samples with unknown province get `-1` and are excluded from the loss.

### MoE (Mixture of Experts) — optional
Replaces selected FFN blocks with a sparse routing mechanism:

```
token (1, D)
    ↓ gate Linear(D, E) → softmax
router probs (1, E)
    ↓ top-2 selection
selected experts {e1, e2} with weights {w1, w2}
    ↓
output = w1 * Expert_e1(token) + w2 * Expert_e2(token)
```

Load-balance aux loss: `MSE(mean_routing_probs, uniform) * E`. This prevents all tokens routing to one expert.

---

## 6. Dataset

### ViMD (Vietnamese Multi-Dialect Dataset)
- **Size:** 102.56 hours, ~19,000 utterances, 1.2M+ words
- **Coverage:** All 63 former Vietnamese provinces
- **Metadata per sample:** speaker ID, gender, province code, province name, audio waveform, text transcript
- **Format:** Multiple `.parquet` files (103 files for the train split)

### Parquet Schema
```
audio:           dict{"array": List[float], "sampling_rate": int}
                 OR dict{"bytes": bytes}
                 OR dict{"path": str}
text:            str      (transcript)
province_name:   str      (e.g., "Hà Nội", "Đà Nẵng")
gender:          str      ("male" / "female")
speaker_id:      str
province_code:   str
```

### Data Placement
```
data/
└── train/
    ├── train-00000-of-00103.parquet
    ├── train-00001-of-00103.parquet
    ├── ...
    └── train-00102-of-00103.parquet
```

### Splits (applied at runtime, no files copied)
- Train: 90% of total rows
- Validation: 5%
- Test: 5%

---

## 7. Setup & Installation

### Requirements
- Python >= 3.11
- CUDA >= 12.4
- NVIDIA GPU with >= 16 GB VRAM (32 GB recommended for `whisper-large-v3` at batch=32)

### Install

```bash
# Clone the repo
git clone <repo-url>
cd Code

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention 2 (requires nvcc / CUDA toolkit)
pip install flash-attn --no-build-isolation
```

> If `flash-attn` installation fails (no CUDA toolkit on the machine), set `model.flash_attention: false` in `configs/config.yaml`. The model will fall back to PyTorch SDPA, which is still faster than vanilla attention.

### W&B Login (optional but recommended)
```bash
wandb login
```

---

## 8. Training

### Quick Start
```bash
python train.py
```

### With CLI Overrides
```bash
# Smaller model for testing
python train.py --model openai/whisper-small --batch_size 8 --lr 1e-4

# Enable MoE
python train.py --moe_enabled true

# Higher LoRA rank
python train.py --lora_r 128

# Resume from checkpoint
python train.py --resume_from outputs/checkpoints/last.ckpt

# Freeze encoder (faster, slightly lower accuracy)
python train.py --freeze_encoder true
```

### Output Files
```
outputs/
├── checkpoints/
│   ├── whisper-vi-00-0.1234.ckpt   # best checkpoints by val_wer
│   ├── whisper-vi-01-0.1089.ckpt
│   └── last.ckpt
└── logs/
    └── whisper-vi-finetune/
        └── ...
```

---

## 9. Evaluation

```bash
# Evaluate from Lightning checkpoint
python evaluate.py --checkpoint outputs/checkpoints/last.ckpt

# With dialect breakdown
python evaluate.py --checkpoint outputs/checkpoints/last.ckpt --dialect_analysis

# From raw LoRA adapter (after merge_and_unload)
python evaluate.py \
    --adapter_path outputs/lora_adapter \
    --base_model openai/whisper-large-v3 \
    --dialect_analysis

# Save results to custom path
python evaluate.py --checkpoint outputs/checkpoints/last.ckpt --output results/test_v1.json
```

### Output Format (`eval_results.json`)
```json
{
  "overall_wer": 0.0892,
  "overall_cer": 0.0341,
  "num_samples": 701,
  "dialect": {
    "Northern": {"wer": 0.072, "cer": 0.028, "count": 312},
    "Central":  {"wer": 0.118, "cer": 0.045, "count": 187},
    "Southern": {"wer": 0.081, "cer": 0.031, "count": 202}
  }
}
```

---

## 10. Configuration Reference

```yaml
model:
  name: "openai/whisper-large-v3"  # any openai/whisper-* variant
  language: "vi"
  task: "transcribe"
  freeze_encoder: false            # freeze encoder, only train decoder LoRA
  freeze_embed: false              # freeze token + position embeddings
  gradient_checkpointing: false    # recompute activations (saves VRAM, ~30% slower)
  flash_attention: true            # Flash Attention 2 (requires flash-attn package)
  torch_compile: false             # torch.compile (experimental with PEFT)

lora:
  enabled: true
  r: 64                            # rank; 8-128 typical range
  alpha: 128                       # effective scale = alpha/r = 2.0
  dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, out_proj, fc1, fc2]
  bias: "none"                     # none | all | lora_only

dialect_adapter:
  enabled: true                    # DAAT on/off
  num_dialects: 3                  # Northern / Central / Southern
  hidden_dim: 256
  dropout: 0.1
  aux_loss_weight: 0.1             # λ — tune via ablation

moe:
  enabled: false
  num_experts: 8
  top_k: 2
  capacity_factor: 1.25
  load_balance_weight: 0.01
  apply_to: "decoder"              # decoder | encoder | both
  replace_every_n: 2               # replace FFN in every N-th layer

data:
  train_data_dir: "data/train"
  val_split: 0.05
  test_split: 0.05
  num_workers: 8                   # 0 for Colab, 8 for dedicated GPU
  pin_memory: true                 # false for Colab
  prefetch_factor: 4
  cache_files: 8                   # parquet files kept in each worker's RAM
  audio_column: "audio"
  text_column: "text"
  province_column: "province_name"

training:
  batch_size: 32                   # reduce if VRAM is insufficient
  gradient_accumulation_steps: 1   # effective batch = batch_size * this
  max_epochs: 10
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  bf16: true
  gradient_clip_val: 1.0
  val_check_interval: 0.25         # validate every 25% of an epoch
  save_top_k: 3
  early_stopping_patience: 5

optimizer:
  scheduler: "cosine"              # cosine | linear | constant
  betas: [0.9, 0.98]              # Whisper paper values
  eps: 1.0e-6

hardware:
  accelerator: "gpu"
  devices: 1
  strategy: "auto"                 # ddp for multi-GPU
  precision: "bf16-mixed"
```

---

## 11. Technical Decisions Log

This section documents **why** each key decision was made, so future maintainers (and LLMs) understand the reasoning.

| Decision | What | Why |
|---|---|---|
| `whisper-large-v3` | Upgraded from `whisper-small` | 32 GB VRAM makes large-v3 practical. Larger model = better Vietnamese phoneme coverage, especially for tonal distinctions. |
| BF16 base weights | `torch_dtype=torch.bfloat16` in `from_pretrained` | Saves ~3 GB VRAM for large-v3 (frozen weights only need inference precision). LoRA adapters are cast back to FP32. |
| LoRA FP32 adapter weights | Cast after `get_peft_model()` | PEFT initializes adapters in the model's dtype (BF16 if loaded in BF16). Training in BF16 can cause instability with small rank updates. FP32 adapters with BF16 base is the correct pattern. |
| Flash Attention 2 | `attn_implementation="flash_attention_2"` | RTX 5090 (Blackwell) supports FA2 natively. ~2-3× speedup for attention computation, especially at sequence length=1500 (encoder) and up to 448 (decoder). |
| LoRA applied before gradient checkpointing | Order in `whisper_lora.py` | PEFT requires the model to be wrapped before checkpointing is enabled. Enabling checkpointing first then wrapping with PEFT causes silent crashes (`use_reentrant=True` incompatibility). |
| `use_reentrant=False` | Arg to `gradient_checkpointing_enable()` | Required for PEFT compatibility. `use_reentrant=True` (PyTorch default) can deadlock with PEFT's custom forward hooks. |
| Gradient checkpointing disabled | `config.yaml` | RTX 5090 has 32 GB VRAM. Gradient checkpointing saves memory at the cost of 30% throughput. Not needed here. |
| `num_workers=8` | DataLoader | Each worker is an independent process with its own LRU parquet file cache. 8 workers saturate the disk I/O without contention. |
| `persistent_workers=True` | DataLoader | Without this, worker processes restart each epoch, losing their warm LRU cache. Persistent workers keep files in RAM across epochs. |
| `drop_last=True` (training only) | DataLoader | Prevents a smaller final batch from skewing batch-norm statistics or causing shape mismatches. No effect with BF16 + Flash Attn but good practice. |
| LRU file cache (not row-group) | `_LRUFileCache` in dataset.py | On a machine with ample RAM, caching full files (after column projection) gives much better cache hit rates than single-row-group caching, because samples are shuffled across all files. |
| Full-batch generation in validation | `lightning_module.py` | Old code looped 1 sample at a time to avoid Colab OOM. RTX 5090 can generate on a full batch of 32 with `num_beams=5` without issue. |
| `encoder_last_hidden_state` for DAAT | From `Seq2SeqLMOutput` | The encoder output is already computed in the main forward pass and returned in `outputs.encoder_last_hidden_state`. Reusing it for the dialect classifier costs zero extra FLOPs. |
| MoE flattened to `(B*T, D)` | `moe.py` | Avoids an outer Python loop over the batch dimension. GPU executes one kernel per expert per top-k slot instead of one per token, improving utilization. |
| `num_sanity_val_steps=1` | `train.py` | Re-enabled after disabling on Colab. Catches config/shape bugs before the first real epoch. |
| `deterministic=False` | `pl.Trainer` | Flash Attention 2 and `torch.compile` are non-deterministic by design. Setting `deterministic=True` would disable them. |
| `β2=0.98, ε=1e-6` | AdamW | Whisper's original training hyperparameters. Higher β2 gives more stable gradient estimates for large models. |

---

## 12. Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >=2.3.0 | Core deep learning framework |
| `torchaudio` | >=2.3.0 | Audio utilities |
| `flash-attn` | >=2.6.0 | Flash Attention 2 kernel (install separately) |
| `transformers` | >=4.44.0 | Whisper model, processor, scheduler |
| `peft` | >=0.12.0 | LoRA via `LoraConfig` + `get_peft_model` |
| `accelerate` | >=0.33.0 | Backend for multi-GPU / precision handling |
| `pytorch-lightning` | >=2.3.0 | Training loop, checkpointing, logging |
| `pyarrow` | >=16.0.0 | Parquet reading with column projection |
| `soundfile` | >=0.12.1 | Audio decoding from bytes |
| `jiwer` | >=3.0.3 | WER and CER computation |
| `wandb` | >=0.17.0 | Experiment tracking |
| `gradio` | >=4.36.0 | Web demo |

---

## Supported Whisper Models

| Size | Parameters | Recommended Batch | Min VRAM |
|---|---|---|---|
| `openai/whisper-tiny` | 39 M | 64 | 4 GB |
| `openai/whisper-base` | 74 M | 64 | 4 GB |
| `openai/whisper-small` | 244 M | 32 | 8 GB |
| `openai/whisper-medium` | 769 M | 16 | 16 GB |
| `openai/whisper-large-v3` | 1550 M | 32 | 28 GB |

**Default config targets `whisper-large-v3` on RTX 5090 (32 GB).** To run on smaller hardware, reduce model size and batch size accordingly, and set `model.flash_attention: false`, `training.bf16: false`, `training.fp16: true`, `data.num_workers: 0`, `data.pin_memory: false`.

---

*Last updated: February 2026 — RTX 5090 refactor with DAAT research contribution.*
