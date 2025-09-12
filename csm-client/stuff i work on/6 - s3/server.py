import os

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_DISABLE_CUDA_GRAPHS": "1",
    }
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import threading
import json
import torch

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 128

import torchaudio
import re
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from generator import load_csm_1b_local, Segment
import warnings
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import base64

warnings.filterwarnings("ignore")

# ----- Configuration -----
NUM_MODEL_WORKERS = 2  # Number of parallel MODEL instances (not threads)
# Be careful: More models = more GPU memory! Start with 2-3

# ----- Globals -----
config = None
model_workers = []  # List of (generator, lock) pairs
voice_segments = {}

current_task = {"text": "", "index": 0, "total": 0}

# ----- FastAPI -----
app = FastAPI()


class VoiceConfig(BaseModel):
    id: int
    name: str
    reference_audio: str
    reference_text: str


class Config(BaseModel):
    model_path: str
    voices: list[VoiceConfig]
    default_voice_id: int = 0


class GenerateRequest(BaseModel):
    texts: List[str]
    voice_id: int = 0


class GenerateResponse(BaseModel):
    success: bool
    message: str
    audio_data: List[str]  # Base64 encoded audio data
    sample_rate: int


def preprocess_text_for_tts(text):
    text = re.sub(r'[“”"]', "", text)
    text = re.sub(r"\s+-\s+", ", ", text)
    text = re.sub(r"(\w)-(\w)", r"\1 \2", text)
    pattern = r"[^\w\s.,!?\']"
    cleaned_text = re.sub(pattern, "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = re.sub(r"([.,!?])(\S)", r"\1 \2", cleaned_text)
    return cleaned_text.strip()


async def generate_with_model_worker(text, voice_id, index, total):
    """Get a model worker and generate audio"""
    # Update current task
    current_task["text"] = text[:50] + "..." if len(text) > 50 else text
    current_task["index"] = index + 1
    current_task["total"] = total

    # Find an available model worker
    for i, (generator, model_lock) in enumerate(model_workers):
        if model_lock.acquire(blocking=False):  # Try to get this model
            try:
                print(f"[Model Worker {i}] Processing text {index+1}/{total}")

                # Preprocess and generate
                cleaned_text = preprocess_text_for_tts(text.lower())
                audio = generator.generate(
                    text=cleaned_text,
                    speaker=0,
                    context=voice_segments[voice_id],
                    max_audio_length_ms=60000,
                    temperature=0.7,
                    topk=30,
                )

                # Convert to base64 instead of saving
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    audio.cpu().unsqueeze(0),
                    generator.sample_rate,
                    format="wav",
                )
                buffer.seek(0)

                audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

                print(f"[Model Worker {i}] Generated audio data for text {index+1}")
                return audio_base64

            except Exception as e:
                print(f"[Model Worker {i}] Error: {e}")
                return None
            finally:
                model_lock.release()

    # If no model available (shouldn't happen with proper queuing)
    return None


def load_reference_segments():
    global voice_segments
    voice_segments = {}

    for voice in config.voices:
        segments = []
        if os.path.isfile(voice.reference_audio):
            wav, sr = torchaudio.load(voice.reference_audio)
            wav = torchaudio.functional.resample(
                wav.squeeze(0), orig_freq=sr, new_freq=24000
            )
            segments.append(Segment(text=voice.reference_text, speaker=0, audio=wav))
        voice_segments[voice.id] = segments


# ----- API Endpoints -----
@app.post("/generate", response_model=GenerateResponse)
async def generate_audio(request: GenerateRequest):
    """Generate audio for all texts using multiple model instances"""
    if not model_workers:
        raise HTTPException(status_code=503, detail="Models not ready yet")

    if request.voice_id not in voice_segments:
        raise HTTPException(
            status_code=400, detail=f"Voice ID {request.voice_id} not available"
        )

    print(
        f"[Server] Generating {len(request.texts)} audio clips with {NUM_MODEL_WORKERS} model workers..."
    )

    # Reset task tracker
    current_task["text"] = ""
    current_task["index"] = 0
    current_task["total"] = len(request.texts)

    # Process all texts - each will grab an available model
    tasks = []
    for i, text in enumerate(request.texts):
        task = generate_with_model_worker(text, request.voice_id, i, len(request.texts))
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Get successful audio data
    successful_audio = [audio for audio in results if audio is not None]

    # Get sample rate from first model
    sample_rate = model_workers[0][0].sample_rate if model_workers else 24000

    return GenerateResponse(
        success=True,
        message=f"Generated {len(successful_audio)}/{len(request.texts)} audio clips using {NUM_MODEL_WORKERS} model workers",
        audio_data=successful_audio,
        sample_rate=sample_rate,
    )


@app.get("/status")
async def get_status():
    """Get current task status"""
    return {
        "current_task": current_task["text"],
        "progress": f"{current_task['index']}/{current_task['total']}",
        "model_workers": NUM_MODEL_WORKERS,
        "active_models": sum(1 for _, lock in model_workers if lock.locked()),
    }


# ----- Server Startup -----
@app.on_event("startup")
async def startup_event():
    global config, model_workers

    print(f"[Startup] Loading {NUM_MODEL_WORKERS} model instances...")

    # Load config
    with open("config_selection.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    # Load multiple model instances
    for i in range(NUM_MODEL_WORKERS):
        try:
            print(f"[Startup] Loading model instance {i+1}/{NUM_MODEL_WORKERS}...")
            generator = load_csm_1b_local(config.model_path, "cuda")
            model_lock = threading.Lock()
            model_workers.append((generator, model_lock))
            print(f"[Startup] Model instance {i+1} loaded successfully")
        except Exception as e:
            print(f"[Startup] Failed to load model instance {i+1}: {e}")
            break

    if not model_workers:
        raise RuntimeError("No model instances could be loaded!")

    # Load reference audio (shared across all models)
    load_reference_segments()

    # Quick warm-up for each model
    warmup_text = "warm-up"
    for i, (generator, lock) in enumerate(model_workers):
        with lock:
            try:
                for voice_id, segments in voice_segments.items():
                    if segments:
                        generator.generate(
                            text=warmup_text,
                            speaker=0,
                            context=segments,
                            max_audio_length_ms=1000,
                            temperature=0.7,
                            topk=30,
                        )
                print(f"[Startup] Model instance {i} warmed up")
            except Exception as e:
                print(f"[Startup] Warm-up failed for model {i}: {e}")

    print(f"[Startup] Ready! {len(model_workers)} model instances loaded")


@app.on_event("shutdown")
async def shutdown_event():
    # Clean up model instances
    for generator, lock in model_workers:
        del generator
    model_workers.clear()
    torch.cuda.empty_cache()


@app.get("/")
def root():
    return {
        "status": "ready",
        "model_instances": len(model_workers),
        "total_workers": NUM_MODEL_WORKERS,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
