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
voice_segments = {}
last_chunk_rms = None
smoothing_factor = 0.3

# ----- FastAPI -----
app = FastAPI()


class VoiceConfig(BaseModel):
    id: int
    name: str
    reference_audio: str
    reference_text: str
    reference_audio2: Optional[str] = None
    reference_text2: Optional[str] = None
    reference_audio3: Optional[str] = None
    reference_text3: Optional[str] = None


# ----- Config Model -----
class Config(BaseModel):
    model_path: str
    voices: list[VoiceConfig]
    default_voice_id: int = 0


# ----- Modified Model Worker Thread -----
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
    for voice_id, segments in voice_segments.items():
        if segments:
            print(f"[Worker] Warming up with voice {voice_id}...", flush=True)
            for chunk in generator.generate_stream(
                text=warmup_text,
                speaker=0,
                context=segments,
                max_audio_length_ms=1000,
                temperature=0.6,
                topk=100,
            ):
                pass

    # SET MODEL.READY
    model_ready.set()
    print("[Worker] Model warm-up complete!", flush=True)

    # Main worker loop
    while model_thread_running.is_set():
        try:
            request = model_queue.get(timeout=0.1)
            if request is None:  # Shutdown signal
                break

            text, speaker_id, context, max_ms, temperature, topk = request

            print(f"[Worker] Processing text: {text[:50]}...")

            # Generate audio stream
            chunk_count = 0
            for chunk in generator.generate_stream(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_ms,
                temperature=temperature,
                topk=topk,
            ):
                chunk_count += 1
                model_result_queue.put(chunk)
                if not model_thread_running.is_set():
                    break

            print(f"[Worker] Generated {chunk_count} chunks for text")
            model_result_queue.put(None)  # EOS marker

            # Insert pause after sentences (1 second of silence)
            if text.strip().endswith((".", "?", "!")):
                silence = torch.zeros(int(generator.sample_rate * 1.0))  # 1s of silence
                model_result_queue.put(silence)
                model_result_queue.put(None)  # EOS marker for silence block

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] Error: {e}", flush=True)
            model_result_queue.put(Exception(f"Generation error: {e}"))
            # Clear any remaining items from the queue to prevent blocking
            while not model_queue.empty():
                try:
                    model_queue.get_nowait()
                except queue.Empty:
                    break

    print("[Worker] Model worker thread exiting", flush=True)


def preprocess_text_for_tts(text):
    print("[DEBUG] Original length (chars):", len(text))
    # Remove all types of quotes
    text = re.sub(r'[“”"]', "", text)

    # Replace " - " with ", "
    text = re.sub(r"\s+-\s+", ", ", text)

    # Replace "-" (between words) with space
    text = re.sub(r"(\w)-(\w)", r"\1 \2", text)

    # Keep only words, spaces, commas, periods, apostrophes, question marks, exclamations
    text = re.sub(r"[^\w\s,.'?!]", "", text)

    # Ensure space after sentence-ending punctuation
    text = re.sub(r"([.?!])(\S)", r"\1 \2", text)

    text = re.sub(r"\s+", " ", text)

    # print("[DEBUG] Cleaned length (chars):", len(text))
    print("[DEBUG] Cleaned text:", text, flush=True)
    return text.strip()


# Add this function to your server code
def split_long_text(text, max_words=150):
    # Split by newlines to get paragraphs
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)

    return chunks


def normalize_with_smoothing(audio_chunk, target_loudness=-16.0):
    global last_chunk_rms

    audio_array = audio_chunk.copy()
    rms = np.sqrt(np.mean(audio_array**2))

    if rms < 1e-6:
        last_chunk_rms = None
        return audio_array

    if last_chunk_rms is not None:
        # Smooth the RMS transition between chunks
        smoothed_rms = smoothing_factor * last_chunk_rms + (1 - smoothing_factor) * rms
        rms = smoothed_rms

    current_loudness = 20 * np.log10(rms)
    gain_db = target_loudness - current_loudness
    gain_linear = 10 ** (gain_db / 20)

    audio_array = audio_array * gain_linear
    last_chunk_rms = rms  # Store for next chunk

    # Prevent clipping
    peak = np.max(np.abs(audio_array))
    if peak > 0.95:
        audio_array = audio_array * (0.95 / peak)
        last_chunk_rms = last_chunk_rms * (0.95 / peak)

    return audio_array


# ----- Modified Audio Generation Function -----
async def audio_generation(text: str, voice_id: int, websocket: WebSocket):
    global is_speaking, voice_segments

    print(f"[DEBUG] Starting audio generation for voice {voice_id}", flush=True)

    if not audio_gen_lock.acquire(blocking=False):
        await websocket.send_json(
            {"type": "error", "message": "Audio generation busy, please wait"}
        )
        return

    try:
        is_speaking = True

        # Clear any previous results from the queue
        while not model_result_queue.empty():
            try:
                model_result_queue.get_nowait()
            except queue.Empty:
                break

        await websocket.send_json({"type": "audio_status", "status": "generating"})

        # Preprocess text
        print(f"[Preprocessing] Original text length: {len(text)}")
        text = preprocess_text_for_tts(text.lower())
        print(f"[Preprocessing] Cleaned text length: {len(text)}")
        voice_context = voice_segments.get(voice_id, [])

        # Split long text into chunks
        text_chunks = split_long_text(text)
        print(f"[Preprocessing] Split into {len(text_chunks)} chunks")

        if len(text_chunks) > 1:
            await websocket.send_json(
                {
                    "type": "status",
                    "message": f"Long text detected. Splitting into {len(text_chunks)} parts...",
                }
            )

        total_chunks_processed = 0

        voice_context = voice_segments.get(voice_id, [])
        print(
            f"[DEBUG] Using voice ID: {voice_id}, context segments: {len(voice_context)}"
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
            avg_wpm = 80
            words_per_second = avg_wpm / 60
            padding_seconds = 2.0  # Reduced from 2000 to 2.0 seconds
            estimated_seconds = len(words) / words_per_second
            max_audio_length_ms = int((estimated_seconds + padding_seconds) * 1000)

            # Send request to model thread
            model_queue.put(
                (
                    chunk,
                    0,
                    voice_segments[voice_id],
                    max_audio_length_ms,
                    0.6,  # temperature
                    100,  # topk
                )
            )

            chunk_counter = 0
            first_chunk_sent = False

            # Stream audio chunks in real-time
            while True:
                try:
                    result = model_result_queue.get(timeout=5.0)  # Increased timeout

                    if result is None:  # End of stream for this chunk
                        break

                    if isinstance(result, Exception):
                        raise result

                    chunk_counter += 1
                    total_chunks_processed += 1

                    # Convert to numpy and send
                    chunk_array = result.cpu().numpy().astype(np.float32)

                    # Normalize with cross-chunk smoothing
                    # chunk_array = normalize_with_smoothing(
                    #     chunk_array, target_loudness=-16.0
                    # )

                    # Check if audio is not silent
                    if np.max(np.abs(chunk_array)) < 0.01:  # Almost silent
                        print(f"[Warning] Chunk {chunk_counter} is almost silent")

                    # Apply gain boost
                    # gain = 1.5
                    # chunk_array = np.clip(chunk_array * gain, -1.0, 1.0)

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
                    print(f"[Warning] Timeout waiting for chunk {chunk_counter}")
                    break

        # All chunks processed
        await websocket.send_json(
            {"type": "status", "message": "All parts processed successfully"}
        )

        await websocket.send_json(
            {
                "type": "completion",
                "message": "audio_generation_complete",
                "chunks_processed": total_chunks_processed,
            }
        )

    except Exception as e:
        print(f"[Audio Generation] Error: {e}", flush=True)
        await websocket.send_json(
            {"type": "error", "message": f"Generation failed: {str(e)}"}
        )

        # Clear queues on error
        while not model_result_queue.empty():
            try:
                model_result_queue.get_nowait()
            except queue.Empty:
                break

    finally:
        is_speaking = False
        audio_gen_lock.release()


# ----- Load Reference Audio -----
def load_reference_segments():
    global voice_segments

    voice_segments = {}
    print(
        f"[DEBUG] Loading reference segments for {len(config.voices)} voices",
        flush=True,
    )

    for voice in config.voices:
        print(f"[DEBUG] Processing voice {voice.id}: {voice.name}", flush=True)
        segments = []

        # Load primary reference (required)
        if os.path.isfile(voice.reference_audio):
            print(
                f"[DEBUG] Loading reference audio: {voice.reference_audio}", flush=True
            )
            wav, sr = torchaudio.load(voice.reference_audio)
            wav = torchaudio.functional.resample(
                wav.squeeze(0), orig_freq=sr, new_freq=24000
            )
            segments.append(Segment(text=voice.reference_text, speaker=0, audio=wav))
            print(f"[DEBUG] Loaded primary reference for voice {voice.id}", flush=True)
        else:
            print(
                f"Warning: Primary reference audio '{voice.reference_audio}' for voice {voice.id} not found",
                flush=True,
            )

        # Load second reference (optional)
        if (
            hasattr(voice, "reference_audio2")
            and voice.reference_audio2
            and os.path.isfile(voice.reference_audio2)
        ):
            print(
                f"[DEBUG] Loading second reference audio: {voice.reference_audio2}",
                flush=True,
            )
            wav, sr = torchaudio.load(voice.reference_audio2)
            wav = torchaudio.functional.resample(
                wav.squeeze(0), orig_freq=sr, new_freq=24000
            )
            segments.append(Segment(text=voice.reference_text2, speaker=0, audio=wav))
            print(f"[DEBUG] Loaded second reference for voice {voice.id}", flush=True)

        # Load third reference (optional)
        if (
            hasattr(voice, "reference_audio3")
            and voice.reference_audio3
            and os.path.isfile(voice.reference_audio3)
        ):
            print(
                f"[DEBUG] Loading third reference audio: {voice.reference_audio3}",
                flush=True,
            )
            wav, sr = torchaudio.load(voice.reference_audio3)
            wav = torchaudio.functional.resample(
                wav.squeeze(0), orig_freq=sr, new_freq=24000
            )
            segments.append(Segment(text=voice.reference_text3, speaker=0, audio=wav))
            print(f"[DEBUG] Loaded third reference for voice {voice.id}", flush=True)

        voice_segments[voice.id] = segments
        print(
            f"[DEBUG] Successfully loaded {len(segments)} segments for voice {voice.id}",
            flush=True,
        )


# ----- Server Startup -----
@app.on_event("startup")
async def startup_event():
    global config, model_thread_running

    print("[Startup] Loading configuration...")

    # Load config from file
    with open("config_selection.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    print("[Startup] Configuration loaded")
    print(f"Model path: {config.model_path}")

    # Load reference audio
    load_reference_segments()

    # Start model worker thread
    model_thread_running.set()
    threading.Thread(target=model_worker, daemon=True, name="model_worker").start()

    print("[Startup] Server ready!")


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
    print("[WebSocket] Client connected")

    # Wait for model to be ready
    if not model_ready.is_set():
        await websocket.send_json(
            {"type": "status", "message": "Models are loading, please wait..."}
        )
        model_ready.wait()

    # Send available voices to client
    voices_info = {voice.id: voice.name for voice in config.voices}
    await websocket.send_json(
        {
            "type": "available_voices",
            "voices": voices_info,
            "default_voice": config.default_voice_id,
        }
    )

    await websocket.send_json(
        {"type": "status", "message": "Models are ready! You can start streaming."}
    )

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "text_message":
                text = data["text"]
                voice_id = data.get("voice_id", config.default_voice_id)
                print(f"[WebSocket] Received text: {text[:50]}...")

                # Validate voice ID
                if voice_id not in voice_segments:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Voice ID {voice_id} not available",
                        }
                    )
                    continue

                # Clear queues before starting new generation
                while not model_result_queue.empty():
                    try:
                        model_result_queue.get_nowait()
                    except queue.Empty:
                        break

                while not model_queue.empty():
                    try:
                        model_queue.get_nowait()
                    except queue.Empty:
                        break

                # Start audio generation
                await audio_generation(text, voice_id, websocket)

                # Send completion message but keep connection open
                await websocket.send_json(
                    {"type": "status", "message": "Ready for next text input"}
                )

    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
        import traceback

        traceback.print_exc()

    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

        # Clear queues on disconnect
        while not model_result_queue.empty():
            try:
                model_result_queue.get_nowait()
            except queue.Empty:
                break

        while not model_queue.empty():
            try:
                model_queue.get_nowait()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9999,
        timeout_keep_alive=300,  # 5 minutes
        timeout_graceful_shutdown=60,  # 1 minute
    )
