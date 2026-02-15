# Optimizing Whisper for Vietnamese Speech Recognition

> **DAT301 Capstone Project**
> Fine-tuning OpenAI Whisper using LoRA + optional MoE for Vietnamese ASR

## Project Structure

```
Code/
├── configs/
│   └── config.yaml              # Main configuration
├── src/
│   ├── data/
│   │   ├── dataset.py           # Parquet dataset loader
│   │   └── datamodule.py        # PyTorch Lightning DataModule
│   ├── models/
│   │   ├── whisper_lora.py      # Whisper + LoRA model builder
│   │   ├── moe.py               # Mixture of Experts module
│   │   └── lightning_module.py  # Lightning training module
│   ├── utils/
│   │   ├── metrics.py           # WER, CER, dialect-wise metrics
│   │   └── text_normalize.py    # Vietnamese text normalization
│   └── demo/
│       └── app.py               # Gradio web demo
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Python dependencies
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Supported Whisper Models

| Size     | Parameters | Model Name                |
|----------|-----------|---------------------------|
| tiny     | 39 M      | `openai/whisper-tiny`     |
| base     | 74 M      | `openai/whisper-base`     |
| small    | 244 M     | `openai/whisper-small`    |
| medium   | 769 M     | `openai/whisper-medium`   |
| large    | 1550 M    | `openai/whisper-large-v3` |

## Data

Place preprocessed parquet files in `data/train/`:
```
data/train/
├── train-00000-of-00103.parquet
├── train-00001-of-00103.parquet
├── ...
└── train-00102-of-00103.parquet
```

## Training

### Basic Training (LoRA only)
```bash
python train.py --config configs/config.yaml
```

### With Custom Model Size
```bash
python train.py --model openai/whisper-medium --batch_size 4
```

### Enable MoE (Mixture of Experts)
```bash
python train.py --moe_enabled true
```

### Resume Training
```bash
python train.py --resume_from outputs/checkpoints/last.ckpt
```

### Key Config Options (configs/config.yaml)
- `model.name`: Whisper model size
- `lora.r`: LoRA rank (higher = more params, default 16)
- `lora.alpha`: LoRA scaling factor (default 32)
- `moe.enabled`: Enable Mixture of Experts
- `moe.num_experts`: Number of experts (default 4)
- `training.batch_size`: Batch size per GPU
- `training.learning_rate`: Learning rate (default 1e-4)

## Evaluation

```bash
# Evaluate from checkpoint
python evaluate.py --checkpoint outputs/checkpoints/last.ckpt

# Evaluate with dialect analysis
python evaluate.py --checkpoint outputs/checkpoints/last.ckpt --dialect_analysis

# Evaluate from saved LoRA adapter
python evaluate.py --adapter_path outputs/lora_adapter --base_model openai/whisper-small
```

## Demo

```bash
# Run Gradio demo
python -m src.demo.app --base_model openai/whisper-small

# With fine-tuned adapter
python -m src.demo.app --model_path outputs/lora_adapter --share
```

## Metrics

- **WER** (Word Error Rate): Primary metric
- **CER** (Character Error Rate): Captures tonal/phonetic errors
- **Dialect-wise WER/CER**: Breakdown by Northern/Central/Southern regions

## Technologies

- PyTorch + PyTorch Lightning
- Hugging Face Transformers + PEFT (LoRA)
- Gradio (Web Demo)
- jiwer (WER/CER computation)
