"""
pipeline.py
------------
CrimeVision-AI core pipeline, refactored into callable functions so both
the CLI script and the Streamlit dashboard can use the same logic.

Stages:
    record_audio()      -> captures microphone input
    transcribe()        -> Whisper speech-to-text
    translate_text()    -> Deep Translator to English
    generate_face()     -> Stable Diffusion composite generation
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionPipeline

from prompt_builder import build_prompt

SAMPLE_RATE = 44100
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"

# Cache heavy models so they load only once per session
_whisper_model = None
_sd_pipe = None


def record_audio(seconds=5, output_file="voice_input.wav", countdown=0):
    """Record microphone audio and save to a WAV file."""
    if countdown:
        time.sleep(countdown)
    recording = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(output_file, SAMPLE_RATE, recording)
    return output_file


def transcribe(audio_file="voice_input.wav"):
    """Convert speech to text using Whisper (multilingual)."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    result = _whisper_model.transcribe(audio_file)
    return result["text"].strip()


def translate_text(text):
    """Translate any supported language to English."""
    if not text:
        return ""
    return GoogleTranslator(source="auto", target="en").translate(text)


def generate_face(prompt_text, output_path="outputs/ai_suspect.png",
                  steps=40, guidance=8):
    """Generate a facial composite from a forensic prompt."""
    global _sd_pipe
    if _sd_pipe is None:
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32
        ).to("cpu")

    image = _sd_pipe(
        prompt_text, num_inference_steps=steps, guidance_scale=guidance
    ).images[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    return output_path


def run_full_pipeline(seconds=5, output_path="outputs/ai_suspect.png"):
    """Convenience wrapper: voice -> text -> translation -> prompt -> image."""
    audio = record_audio(seconds=seconds)
    text = transcribe(audio)
    translated = translate_text(text)
    prompt = build_prompt(translated)
    image_path = generate_face(prompt, output_path=output_path)
    return {
        "transcription": text,
        "translation": translated,
        "prompt": prompt,
        "image_path": image_path,
    }