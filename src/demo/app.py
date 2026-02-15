"""
Gradio Web Demo for Vietnamese Speech-to-Text with fine-tuned Whisper.

Usage:
    python -m src.demo.app
    python -m src.demo.app --model_path outputs/lora_adapter
    python -m src.demo.app --base_model openai/whisper-small --share
"""

import argparse

import torch
import gradio as gr
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from src.utils.text_normalize import normalize_vietnamese


def load_model(base_model: str, adapter_path: str = None):
    """Load Whisper model with optional LoRA adapter."""
    print(f"Loading base model: {base_model}")
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, processor, device


def transcribe(audio, model, processor, device):
    """Transcribe audio input.

    Args:
        audio: Tuple of (sample_rate, numpy_array) from Gradio.
    """
    if audio is None:
        return "No audio provided."

    sr, audio_array = audio
    # Convert to float32 and normalize
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if audio_array.max() > 1.0:
        audio_array = audio_array / 32768.0

    # Mono conversion
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    # Process and transcribe
    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="vi",
            task="transcribe",
        )

    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return transcription


def build_demo(model, processor, device):
    """Build the Gradio interface."""

    def process_audio(audio):
        return transcribe(audio, model, processor, device)

    with gr.Blocks(
        title="Vietnamese Speech-to-Text",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎙️ Vietnamese Speech-to-Text
            ### Powered by Whisper + LoRA Fine-tuning

            Upload an audio file or record using your microphone to get
            Vietnamese transcription.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="numpy",
                    sources=["microphone", "upload"],
                )
                submit_btn = gr.Button(
                    "Transcribe", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Transcription",
                    lines=6,
                    placeholder="Transcription will appear here...",
                )

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[output_text],
        )

        gr.Markdown(
            """
            ---
            **Model:** Whisper fine-tuned with LoRA for Vietnamese |
            **Dataset:** ViMD (63 provincial dialects) |
            **Project:** DAT301 Capstone
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Vietnamese STT Demo")
    parser.add_argument(
        "--base_model", type=str, default="openai/whisper-small"
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model, processor, device = load_model(args.base_model, args.model_path)
    demo = build_demo(model, processor, device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
