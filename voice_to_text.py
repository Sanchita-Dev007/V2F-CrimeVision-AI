import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from deep_translator import GoogleTranslator
import time

fs = 44100
seconds = 5

print("Recording will start in 3 seconds...")
time.sleep(3)

print("Speak now (Hindi / English supported)...")

# record voice
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("audio/voice_input.wav", fs, recording)

# load whisper model
model = whisper.load_model("base")

# speech to text
result = model.transcribe("audio/voice_input.wav")

text = result["text"]

print("Detected text:", text)

# translate to English
translated = GoogleTranslator(
    source='auto',
    target='en'
).translate(text)

print("Translated text:", translated)