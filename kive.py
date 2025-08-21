import os
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_LAUNCH_BLOCKING": "1",
    "PYTORCH_DISABLE_CUDA_GRAPHS": "1",
    })

import threading
import time
import json
import queue
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional
from generator import load_csm_1b_local, Segment
import warnings

warnings.filterwarnings("ignore")

# ----- Globals -----
config = None
generator = None
model_ready = threading.Event()
model_thread_running = threading.Event()
model_queue = queue.Queue()
model_result_queue = queue.Queue()
audio_gen_lock = threading.Lock()
is_speaking = False
active_connections = []
reference_segments = []

# ----- FastAPI -----
app = FastAPI()

# ----- Config Model -----
class Config(BaseModel):
    model_path: str
    voice_speaker_id: int = 0
    reference_audio: str
    reference_text: str
    reference_audio2: Optional[str] = None
    reference_text2: Optional[str] = None
    reference_audio3: Optional[str] = None
    reference_text3: Optional[str] = None

# ----- Model Worker Thread -----
def model_worker():
    global generator

    print("[Worker] Starting model loading...", flush=True)

    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.fx_graph_cache = False

    # Load the voice model
    generator = load_csm_1b_local(config.model_path, "cuda")

    print("[Worker] CSM Model loaded. Starting warm-up...")

    # Warm-up the model
    warmup_text = "warm-up " * 5
    for chunk in generator.generate_stream(
            text=warmup_text,
            speaker=config.voice_speaker_id,
            context=reference_segments,
            max_audio_length_ms=1000,
            temperature=0.7,
            topk=40
            ):
        pass

    # SET MODEL.READY
    model_ready.set()

    print("[Worker] Model warm-up complete!", flush=True)

    # Main worker loop
    while model_thread_running.is_set():
        try:
            request = model_queue.get(timeout=0.1)
            if request is None:
                break

            text, speaker_id, context, max_ms, temperature, topk = request

            # Generate audio stream
            for chunk in generator.generate_stream(
                    text=text,
                    speaker=speaker_id,
                    context=context,
                    max_audio_length_ms=max_ms,
                    temperature=temperature,
                    topk=topk
                    ):
                model_result_queue.put(chunk)
                if not model_thread_running.is_set():
                    break

            model_result_queue.put(None)  # EOS marker

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] Error: {e}", flush=True)
            model_result_queue.put(Exception(f"Generation error: {e}"))

    print("[Worker] Model worker thread exiting", flush=True)

# ----- Audio Generation Function -----
async def audio_generation(text: str, websocket: WebSocket):
    global is_speaking, reference_segments

    if not audio_gen_lock.acquire(blocking=False):
        await websocket.send_json({
            "type": "error",
            "message": "Audio generation busy, please wait"
            })
        return

    try:
        is_speaking = True

        await websocket.send_json({
            "type": "audio_status",
            "status": "generating"
            })

        # ADD PREPROCESSING LATER HERE
        # Estimate audio length for better streaming
        words = text.split()
        avg_wpm = 100
        words_per_second = avg_wpm / 60
        estimated_seconds = len(words) / words_per_second
        max_audio_length_ms = int(estimated_seconds * 1000)

        # Send request to model thread
        model_queue.put((
            text,
            config.voice_speaker_id,
            reference_segments,
            max_audio_length_ms,
            0.8,  # temperature
            50    # topk
            ))

        chunk_counter = 0
        first_chunk_sent = False

        # Stream audio chunks in real-time
        while True:
            try:
                result = model_result_queue.get(timeout=1.0)

                if result is None:  # End of stream
                    break

                if isinstance(result, Exception):
                    raise result

                chunk_counter += 1

                # Convert to numpy and send
                chunk_array = result.cpu().numpy().astype(np.float32)

                # Send first chunk status
                if not first_chunk_sent:
                    await websocket.send_json({
                        "type": "audio_status",
                        "status": "first_chunk"
                        })
                    first_chunk_sent = True

                # Send audio chunk
                await websocket.send_json({
                    "type": "audio_chunk",
                    "audio": chunk_array.tolist(),
                    "sample_rate": generator.sample_rate,
                    "chunk_num": chunk_counter
                    })

            except queue.Empty:
                continue

    except Exception as e:
        print(f"[Audio Generation] Error: {e}", flush=True)
        await websocket.send_json({
            "type": "error",
            "message": f"Generation failed: {str(e)}"
            })

    finally:
        is_speaking = False
        audio_gen_lock.release()

        # Send end of stream
        await websocket.send_json({
            "type": "audio_status",
            "status": "complete"
            })

# ----- Load Reference Audio -----
def load_reference_segments():
    global reference_segments

    reference_segments = []

    # Load primary reference
    if os.path.isfile(config.reference_audio):
        print(f"Loading reference audio: {config.reference_audio}")
        wav, sr = torchaudio.load(config.reference_audio)
        wav = torchaudio.functional.resample(wav.squeeze(0), orig_freq=sr, new_freq=24000)
        reference_segments.append(Segment(
            text=config.reference_text,
            speaker=config.voice_speaker_id,
            audio=wav
            ))
    else:
        print(f"Warning: Reference audio '{config.reference_audio}' not found")

# ----- Server Startup -----
@app.on_event("startup")
async def startup_event():
    global config, model_thread_running

    print("[Startup] Loading configuration...")

    # Load config from file
    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    print("[Startup] Configuration loaded")
    print(f"Model path: {config.model_path}")

    # Load reference audio
    load_reference_segments()

    # Start model worker thread
    model_thread_running.set()
    threading.Thread(
            target=model_worker,
            daemon=True,
            name="model_worker"
            ).start()

    print("[Startup] Server ready!")

# ----- WebSocket Endpoint -----
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    print("[WebSocket] Client connected")

    # Wait for model to be ready
    if not model_ready.is_set():
        await websocket.send_json({
            "type": "status",
            "message": "Models are loading, please wait..."
            })
        model_ready.wait()

    await websocket.send_json({
        "type": "status",
        "message": "Models are ready! You can start streaming."
        })

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "text_message":
                text = data["text"]
                print(f"[WebSocket] Received text: {text}")

                # Start audio generation
                await audio_generation(text, websocket)

    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print("[WebSocket] Client disconnected")


# ----- Root Endpoint -----
@app.get("/")
def root():
    return {
            "status": "ready",
            "model_loaded": model_ready.is_set()
            }

# ----- Health Check -----
@app.get("/health")
def health():
    return {
            "status": "healthy",
            "model_ready": model_ready.is_set(),
            "is_speaking": is_speaking
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
            app,
            host="0.0.0.0",
            port=8443,
            timeout_keep_alive=300,          # 5 minutes
            timeout_graceful_shutdown=60     # 1 minute
            )

