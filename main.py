# Import libraries
import io  # For memory stream
from fastapi import FastAPI
from fastapi.responses import (
    FileResponse,  # Add file response for local file system
    StreamingResponse,  # Add streaming response for memory stream
)
from ml import obtain_image

# Create a FastAPI instance
app = FastAPI()


# Create an app
@app.get("/")
def read_root():
    return {"Hello": "World"}


# Create an app that uses the path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


# Create an app that generates an image
@app.get("/generate")
def generate_image(
    prompt: str,
    *,  # This means that the following arguments are keyword-only
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    image = obtain_image(
        prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    image.save("image.png")
    return FileResponse("image.png")


@app.get("/generate-memory")
def generate_image_memory(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")


# How to run the app
# uvicorn main:app or uvicorn main:app --reload (for development)
