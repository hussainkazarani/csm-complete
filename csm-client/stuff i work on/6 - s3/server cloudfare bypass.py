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
from generator import load_csm_1b_local, Segment, generate_streaming_audio
import warnings
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import base64
import tempfile
from fastapi.responses import StreamingResponse

warnings.filterwarnings("ignore")

# ----- Configuration -----
NUM_MODEL_WORKERS = 2  # Number of parallel MODEL instances (not threads)

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
    """Get a model worker and generate audio using streaming"""
    # Update current task
    current_task["text"] = text[:50] + "..." if len(text) > 50 else text
    current_task["index"] = index + 1
    current_task["total"] = total

    print(f"[Worker] Starting generation for text {index+1}/{total}", flush=True)

    # Find an available model worker
    for i, (generator, model_lock) in enumerate(model_workers):
        if model_lock.acquire(blocking=False):
            try:
                print(
                    f"[Model Worker {i}] Processing text {index+1}/{total}: '{text[:50]}...'",
                    flush=True,
                )

                # Preprocess text
                cleaned_text = preprocess_text_for_tts(text.lower())

                # Create temporary file for streaming output
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_filename = temp_file.name

                # Generate audio using streaming
                generate_streaming_audio(
                    generator=generator,
                    text=cleaned_text,
                    speaker=0,
                    context=voice_segments[voice_id],
                    output_file=temp_filename,
                )

                # Read the generated audio file and convert to base64
                with open(temp_filename, "rb") as f:
                    audio_data = f.read()

                # Clean up temporary file
                os.unlink(temp_filename)

                audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                print(
                    f"[Model Worker {i}] Generated audio for text {index+1}", flush=True
                )
                return audio_base64

            except Exception as e:
                print(f"[Model Worker {i}] Error: {e}", flush=True)
                if "temp_filename" in locals() and os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                return None
            finally:
                model_lock.release()

    print("[Worker] No available model workers, waiting...", flush=True)
    # If no model available, wait a bit and try again
    await asyncio.sleep(0.1)
    return await generate_with_model_worker(text, voice_id, index, total)


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
@app.post("/generate")
async def generate_audio(request: GenerateRequest):
    """Generate and stream audio files one by one"""
    if not model_workers:
        raise HTTPException(status_code=503, detail="Models not ready yet")

    if request.voice_id not in voice_segments:
        raise HTTPException(
            status_code=400, detail=f"Voice ID {request.voice_id} not available"
        )

    print(f"[Server] Starting generation for {len(request.texts)} texts", flush=True)

    async def generate_stream():
        # Process each text one by one and stream immediately
        for i, text in enumerate(request.texts):
            print(
                f"[Server] Working on text {i+1}/{len(request.texts)}: '{text[:50]}...'",
                flush=True,
            )

            audio_base64 = await generate_with_model_worker(
                text, request.voice_id, i, len(request.texts)
            )

            if audio_base64:
                print(f"[Server] Sending audio {i+1} to client", flush=True)
                yield json.dumps(
                    {
                        "index": i,
                        "total": len(request.texts),
                        "audio_data": audio_base64,
                        "status": "success",
                    }
                ) + "\n"
            else:
                print(f"[Server] Failed to generate audio {i+1}", flush=True)
                yield json.dumps(
                    {
                        "index": i,
                        "total": len(request.texts),
                        "status": "error",
                        "message": "Generation failed",
                    }
                ) + "\n"

        print("[Server] All jobs completed", flush=True)

    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")


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

    print(f"[Startup] Loading {NUM_MODEL_WORKERS} model instances...", flush=True)

    # Load config
    with open("config_selection.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    # Load multiple model instances
    for i in range(NUM_MODEL_WORKERS):
        try:
            print(
                f"[Startup] Loading model instance {i+1}/{NUM_MODEL_WORKERS}...",
                flush=True,
            )
            generator = load_csm_1b_local(config.model_path, "cuda")
            model_lock = threading.Lock()
            model_workers.append((generator, model_lock))
            print(f"[Startup] Model instance {i+1} loaded successfully", flush=True)
        except Exception as e:
            print(f"[Startup] Failed to load model instance {i+1}: {e}", flush=True)
            break

    if not model_workers:
        raise RuntimeError("No model instances could be loaded!")

    # Load reference audio (shared across all models)
    load_reference_segments()

    # Quick warm-up for each model using streaming
    warmup_text = "warm-up"
    for i, (generator, lock) in enumerate(model_workers):
        with lock:
            try:
                for voice_id, segments in voice_segments.items():
                    if segments:
                        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                            generate_streaming_audio(
                                generator=generator,
                                text=warmup_text,
                                speaker=0,
                                context=segments,
                                output_file=temp_file.name,
                            )
                print(
                    f"[Startup] Model instance {i} warmed up with streaming", flush=True
                )
            except Exception as e:
                print(f"[Startup] Warm-up failed for model {i}: {e}", flush=True)

    print(
        f"[Startup] Ready! {len(model_workers)} model instances loaded with streaming support",
        flush=True,
    )


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
        "streaming_support": True,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
