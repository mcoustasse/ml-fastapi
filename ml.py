# Import libraries
# from pathlib import Path
from dotenv import load_dotenv
import os

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

# Load token from .env file
load_dotenv()
token = os.getenv("huggingface_token")

# token_path = Path("token.txt")
# token = token_path.read_text().strip()

# Load pipeline (vscode might complain about the type or words, but it's fine)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=token,
)

# Lets move it to GPU
pipe.to("mps")  # or "cuda" if you have a GPU

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()


# Function to generate an image
def obtain_image(
    prompt: str,
    *,  # This means that the following arguments are keyword-only
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image:
    # First-time "warmup" pass
    # (see explanation: https://huggingface.co/docs/diffusers/optimization/mps)
    _ = pipe(prompt, num_inference_steps=1)
    print("Warmup pass complete. Starting inference...")
    # Generator for reproducibility
    generator = (
        None if seed is None else torch.Generator().manual_seed(seed)
    )  # or torch.Generator(device="cuda") if you have a GPU
    print(f"Using device: {pipe.device}")
    # Generate image
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image
