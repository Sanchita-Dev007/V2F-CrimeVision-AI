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

import tkinter as tk
from PIL import Image, ImageTk


fs = 44100
seconds = 5


print("Loading models...")

model = whisper.load_model("base")

pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

print("Models loaded")


def generate_face():

    status_label.config(text="Recording...")
    root.update()

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()

    write("voice_input.wav", fs, recording)

    status_label.config(text="Processing...")
    root.update()

    result = model.transcribe("voice_input.wav")

    text = result["text"]

    translated = GoogleTranslator(
        source="auto",
        target="en"
    ).translate(text)

    prompt = build_prompt(translated)

    status_label.config(text="Generating...")
    root.update()

    image = pipe(
        prompt,
        num_inference_steps=40,
        guidance_scale=8
    ).images[0]

    os.makedirs("outputs", exist_ok=True)

    path = "outputs/gui_face.png"

    image.save(path)

    img = Image.open(path)
    img = img.resize((256, 256))

    img = ImageTk.PhotoImage(img)

    image_label.config(image=img)
    image_label.image = img

    status_label.config(text="Done")


root = tk.Tk()
root.title("CrimeVision AI")

btn = tk.Button(
    root,
    text="Record & Generate",
    command=generate_face,
    height=2,
    width=20
)

btn.pack()

status_label = tk.Label(root, text="Ready")
status_label.pack()

image_label = tk.Label(root)
image_label.pack()

root.mainloop()