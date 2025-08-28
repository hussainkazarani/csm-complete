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
import time
import json
import queue
import torch
import torchaudio
import re
import numpy as np
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional
from generator import load_csm_1b_local, Segment
from llm_interface import LLMInterface
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
llm = None
llm_lock = threading.Lock()
conversation_history = []

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
    llm_path: str
    system_prompt: str = "You are a helpful AI assistant."
    max_tokens: int = 100


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
        topk=40,
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
                topk=topk,
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


def preprocess_text_for_tts(text):
    # This includes: ; : " '  ~ @ # $ % ^ & * ( ) _ - + = [ ] { } \ | / < >
    pattern = r"[^\w\s.,!?\']"
    # Replace matched punctuation with empty string
    cleaned_text = re.sub(pattern, "", text)
    # normalize multiple spaces to single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # ensure there's a space after punctuation for better speech pacing
    cleaned_text = re.sub(r"([.,!?])(\S)", r"\1 \2", cleaned_text)
    return cleaned_text.strip()


# Add this function to your server code
def split_long_text(text, max_words=100):
    """
    Split long text into chunks that the TTS model can handle
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)

    return chunks


# ----- Modified Audio Generation Function -----
async def audio_generation(text: str, websocket: WebSocket):
    global is_speaking, reference_segments

    if not audio_gen_lock.acquire(blocking=False):
        await websocket.send_json(
            {"type": "error", "message": "Audio generation busy, please wait"}
        )
        return

    try:
        is_speaking = True

        await websocket.send_json({"type": "audio_status", "status": "generating"})

        # Preprocess text
        print(f"[Preprocessing] Original text length: {len(text)}")
        text = preprocess_text_for_tts(text.lower())
        print(f"[Preprocessing] Cleaned text length: {len(text)}")

        # Split long text into chunks
        text_chunks = split_long_text(text, max_words=80)
        print(f"[Preprocessing] Split into {len(text_chunks)} chunks")

        if len(text_chunks) > 1:
            await websocket.send_json(
                {
                    "type": "status",
                    "message": f"Long text detected. Splitting into {len(text_chunks)} parts...",
                }
            )

        # Process each chunk
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                await websocket.send_json(
                    {
                        "type": "status",
                        "message": f"Processing part {i+1}/{len(text_chunks)}...",
                    }
                )

            # Estimate audio length for better streaming
            words = chunk.split()
            avg_wpm = 100
            words_per_second = avg_wpm / 60
            estimated_seconds = len(words) / words_per_second
            max_audio_length_ms = int(estimated_seconds * 1000)

            # Send request to model thread
            model_queue.put(
                (
                    chunk,
                    config.voice_speaker_id,
                    reference_segments,
                    max_audio_length_ms,
                    0.8,  # temperature
                    50,  # topk
                )
            )

            chunk_counter = 0
            first_chunk_sent = False

            # Stream audio chunks in real-time
            while True:
                try:
                    result = model_result_queue.get(timeout=1.0)

                    if result is None:  # End of stream for this chunk
                        break

                    if isinstance(result, Exception):
                        raise result

                    chunk_counter += 1

                    # Convert to numpy and send
                    chunk_array = result.cpu().numpy().astype(np.float32)

                    # Send first chunk status
                    if not first_chunk_sent:
                        await websocket.send_json(
                            {"type": "audio_status", "status": "first_chunk"}
                        )
                        first_chunk_sent = True

                    # Send audio chunk
                    await websocket.send_json(
                        {
                            "type": "audio_chunk",
                            "audio": chunk_array.tolist(),
                            "sample_rate": generator.sample_rate,
                            "chunk_num": chunk_counter,
                            "part": (
                                f"{i+1}/{len(text_chunks)}"
                                if len(text_chunks) > 1
                                else None
                            ),
                        }
                    )

                except queue.Empty:
                    continue

        # All chunks processed
        await websocket.send_json(
            {"type": "status", "message": "All parts processed successfully"}
        )

    except Exception as e:
        print(f"[Audio Generation] Error: {e}", flush=True)
        await websocket.send_json(
            {"type": "error", "message": f"Generation failed: {str(e)}"}
        )

    finally:
        is_speaking = False
        audio_gen_lock.release()

        # Send end of stream
        await websocket.send_json({"type": "audio_status", "status": "complete"})


# ----- Load Reference Audio -----
def load_reference_segments():
    global reference_segments

    reference_segments = []

    # Load primary reference
    if os.path.isfile(config.reference_audio):
        print(f"Loading reference audio: {config.reference_audio}")
        wav, sr = torchaudio.load(config.reference_audio)
        wav = torchaudio.functional.resample(
            wav.squeeze(0), orig_freq=sr, new_freq=24000
        )
        reference_segments.append(
            Segment(
                text=config.reference_text, speaker=config.voice_speaker_id, audio=wav
            )
        )
    else:
        print(f"Warning: Reference audio '{config.reference_audio}' not found")


# ----- Server Startup -----
@app.on_event("startup")
async def startup_event():
    global config, model_thread_running, llm

    print("[Startup] Loading configuration...")

    # Load config from file
    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    print("[Startup] Configuration loaded", flush=True)
    print(f"Model path: {config.model_path}", flush=True)
    print(f"LLM path: {config.llm_path}", flush=True)

    # Load reference audio
    load_reference_segments()

    print("[Startup] Loading LLM...", flush=True)
    llm = LLMInterface(config.llm_path, max_tokens=config.max_tokens)
    print("[Startup] LLM loaded", flush=True)

    # Start model worker thread
    model_thread_running.set()
    threading.Thread(target=model_worker, daemon=True, name="model_worker").start()

    print("[Startup] Server ready!", flush=True)


# ----- WebSocket Endpoint -----
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if len(active_connections) > 0:
        await websocket.send_json(
            {"type": "error", "message": "Server busy with another connection"}
        )
        await websocket.close()
        return

    await websocket.accept()
    active_connections.append(websocket)

    #    print(f"[WebSocket] Client connected. Active connections: {len(active_connections)}")
    #    print(f"[WebSocket] Model queue size: {model_queue.qsize()}")
    #    print(f"[WebSocket] Result queue size: {model_result_queue.qsize()}")

    print("[WebSocket] Client connected")

    # Wait for model to be ready
    if not model_ready.is_set():
        await websocket.send_json(
            {"type": "status", "message": "Models are loading, please wait..."}
        )
        model_ready.wait()

    await websocket.send_json(
        {"type": "status", "message": "Models are ready! You can start streaming."}
    )

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "text_message":
                text = data["text"]
                print(f"[WebSocket] Received text: {text}")

                # Generate LLM response first
                ai_response = await generate_llm_response(text, websocket)
                print(f"[LLM] Response: {ai_response}")

                # Start audio generation
                await audio_generation(ai_response, websocket)

    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

        global conversation_history
        conversation_history = []

        while not model_result_queue.empty():
            try:
                model_result_queue.get_nowait()
            except queue.Empty:
                break

        print("[WebSocket] Client disconnected")


# ----- Root Endpoint -----
@app.get("/")
def root():
    return {"status": "ready", "model_loaded": model_ready.is_set()}


# ----- Health Check -----
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_ready": model_ready.is_set(),
        "is_speaking": is_speaking,
    }


# ----- LLM -----
async def generate_llm_response(user_text, websocket):
    global conversation_history, llm, config

    await websocket.send_json({"type": "status", "message": "Thinking..."})

    try:
        # Format conversation history for LLM
        history_text = ""
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            history_text += f"User: {msg['user']}\nAssistant: {msg['ai']}\n"

        # Generate response with LLM
        with llm_lock:
            ai_response = llm.generate_response(
                config.system_prompt, user_text, history_text
            )

        # Send the LLM response as text before starting TTS
        await websocket.send_json({"type": "llm_response", "text": ai_response})

        # Update conversation history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append(
            {"timestamp": timestamp, "user": user_text, "ai": ai_response}
        )

        # Keep only last 20 messages to prevent memory bloat
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        return ai_response

    except Exception as e:
        print(f"[LLM] Error: {e}", flush=True)
        error_msg = "I'm having trouble thinking right now. Please try again."

        # Send error as LLM response
        await websocket.send_json({"type": "llm_response", "text": error_msg})
        return error_msg


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        timeout_keep_alive=300,  # 5 minutes
        timeout_graceful_shutdown=60,  # 1 minute
    )

