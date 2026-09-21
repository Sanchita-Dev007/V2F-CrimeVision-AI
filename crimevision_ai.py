import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import time

from deep_translator import GoogleTranslator
from prompt_builder import build_prompt

from diffusers import StableDiffusionPipeline
import torch

from history_lookup import match_face


# ---------------- AUDIO RECORD ----------------

fs = 44100
seconds = 5

print("Recording will start in 3 seconds...")
time.sleep(3)

print("Speak now (Hindi / English supported)...")

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("voice_input.wav", fs, recording)


# ---------------- SPEECH TO TEXT ----------------

model = whisper.load_model("base")

result = model.transcribe("voice_input.wav")

text = result["text"]

print("Detected text:", text)


# ---------------- TRANSLATE ----------------

translated = GoogleTranslator(
    source="auto",
    target="en"
).translate(text)

print("Translated text:", translated)


# ---------------- PROMPT BUILDER ----------------

prompt = build_prompt(translated)

print("Final AI prompt:", prompt)


# ---------------- LOAD MODEL ----------------

model_id = "sd-legacy/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")


# ---------------- GENERATE IMAGE ----------------

image = pipe(
    prompt,
    num_inference_steps=40,
    guidance_scale=8
).images[0]


# ---------------- SAVE OUTPUT ----------------

os.makedirs("outputs", exist_ok=True)

output_path = "outputs/ai_suspect.png"
image.save(output_path)

print("Image saved in outputs folder")


# ---------------- CASE HISTORY LOOKUP ----------------

print("\n--- Checking case database for similar records ---")

history_result = match_face(output_path)

print(history_result["message"])

for match in history_result["matches"]:
    print(
        f"  Case ID: {match['case_id']} | "
        f"Crime: {match['crime_type']} | "
        f"Prior FIRs: {match['fir_count']} | "
        f"Status: {match['status']} | "
        f"Last Reported: {match['last_reported']} | "
        f"Similarity: {match['similarity_score']}%"
    )