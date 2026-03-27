from diffusers import StableDiffusionPipeline
import torch

# model to download
model_id = "sd-legacy/stable-diffusion-v1-5"

# load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

# memory optimization (important for 8GB RAM)
pipe.enable_attention_slicing()

# prompt for the AI image
prompt = "police sketch of a criminal suspect, black and white forensic portrait"

# generate image
image = pipe(prompt).images[0]

# save output
image.save("suspect_face.png")

print("Face generated successfully!")