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


# ----- Server Startup -----
@app.on_event("startup")
async def startup_event():
    global config, model_thread_running

    print("[Startup] Loading configuration...", flush=True)

    # Load config from file
    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    print("[Startup] Configuration loaded", flush=True)
    print(f"Model path: {config.model_path}")

    # Load reference audio
    load_reference_segments()

    # Start model worker thread
    model_thread_running.set()
    threading.Thread(target=model_worker, daemon=True, name="model_worker").start()

    print("[Startup] Server ready!", flush=True)


# ----- Load Reference Segments -----
def load_reference_segments():
    global reference_segments
    reference_segments = []

    # load primary reference
    primary_segment = load_single_reference(
        config.reference_audio, config.reference_text, config.voice_speaker_id
    )
    if not primary_segment:
        print("Error: Primary reference audio is required but not found!", flush=True)
        return

    reference_segments.append(primary_segment)

    # load second reference
    if config.reference_audio2 and config.reference_text2:
        segment = load_single_reference(
            config.reference_audio2, config.reference_text2, config.voice_speaker_id
        )
        if segment:
            reference_segments.append(segment)
        else:
            print("Second reference audio is not found!", flush=True)

    # load third reference
    if config.reference_audio3 and config.reference_text3:
        segment = load_single_reference(
            config.reference_audio3, config.reference_text3, config.voice_speaker_id
        )
        if segment:
            reference_segments.append(segment)
        else:
            print("Third reference audio is not found!", flush=True)


# ----- Helper Function To Load Single Reference Audio -----
def load_single_reference(audio_path, text, speaker_id):
    if os.path.isfile(audio_path):
        print(f"Loading reference audio: {audio_path}", flush=True)
        wav, sr = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = torchaudio.functional.resample(
            wav.squeeze(0), orig_freq=sr, new_freq=24000
        )
        return Segment(text=text, speaker=speaker_id, audio=wav)
    else:
        print(f"Warning: Reference audio '{audio_path}' not found", flush=True)
        return None


# ----- Model Worker Thread -----
def model_worker():
    global generator

    print("[Worker] Starting model loading...", flush=True)

    # Optimize torch settings
    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.fx_graph_cache = True
    torch.backends.cudnn.benchmark = True

    # Load the voice model
    generator = load_csm_1b_local(config.model_path, "cuda")

    print("[Worker] CSM Model loaded. Starting warm-up...", flush=True)

    # Better warm-up with varied text
    warmup_texts = [
        "Hello, this is a warm-up sequence.",
        "The quick brown fox jumps over the lazy dog.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
    ]
    
    for warmup_text in warmup_texts:
        try:
            for chunk in generator.generate_stream(
                text=warmup_text,
                speaker=config.voice_speaker_id,
                context=reference_segments,
                max_audio_length_ms=2000,
                temperature=0.7,
                topk=40,
            ):
                pass
            time.sleep(0.5)
        except Exception as e:
            print(f"[Worker] Warm-up error: {e}", flush=True)

    model_ready.set()
    print("[Worker] Model warm-up complete!", flush=True)

    # Main worker loop
    while model_thread_running.is_set():
        try:
            request = model_queue.get(timeout=0.5)
            if request is None:
                break

            text, speaker_id, context, max_ms, temperature, topk = request

            # Generate audio with consistent parameters
            for chunk in generator.generate_stream(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=min(max_ms, 30000),
                temperature=0.7,
                topk=40,
            ):
                if not model_thread_running.is_set():
                    break
                model_result_queue.put(chunk)

            model_result_queue.put("EOS_AUDIO")

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
        await websocket.send_json(
            {"type": "error", "message": "Audio generation busy, please wait"}
        )
        return

    try:
        is_speaking = True

        await websocket.send_json({"type": "audio_status", "status": "generating"})

        # Preprocess text
        print(f"[Preprocessing] Original text: {text}", flush=True)
        text = preprocess_text_for_tts(text)
        print(f"[Preprocessing] Cleaned text: {text}", flush=True)

        # Split text into meaningful chunks
        text_chunks = split_text_into_meaningful_chunks(text, max_words=50)
        print(f"[Preprocessing] Split into {len(text_chunks)} chunks", flush=True)

        if len(text_chunks) > 1:
            await websocket.send_json(
                {
                    "type": "status",
                    "message": f"Text split into {len(text_chunks)} parts for better processing...",
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

            # Calculate appropriate audio length
            words = chunk.split()
            estimated_seconds = max(3, len(words) * 0.8)
            max_audio_length_ms = int(estimated_seconds * 1000) + 2000

            # Clear any previous results
            while not model_result_queue.empty():
                try:
                    model_result_queue.get_nowait()
                except queue.Empty:
                    break

            # Send request to model thread
            model_queue.put(
                (
                    chunk,
                    config.voice_speaker_id,
                    reference_segments,
                    max_audio_length_ms,
                    0.7,
                    40,
                )
            )

            chunk_counter = 0
            audio_chunks = []
            start_time = time.time()

            # Stream audio chunks
            while True:
                try:
                    result = model_result_queue.get(timeout=10.0)

                    if result == "EOS_AUDIO":
                        break

                    elif isinstance(result, Exception):
                        raise result

                    # Collect and process audio chunks
                    else:
                        chunk_array = result.cpu().numpy().astype(np.float32)
                        
                        # Apply gentle normalization
                        max_val = np.max(np.abs(chunk_array))
                        if max_val > 0:
                            chunk_array = chunk_array * (0.9 / max_val)
                        
                        audio_chunks.append(chunk_array)
                        chunk_counter += 1

                        # Send chunks in batches
                        if chunk_counter % 3 == 0 and audio_chunks:
                            combined_chunk = np.concatenate(audio_chunks)
                            await websocket.send_json(
                                {
                                    "type": "audio_chunk",
                                    "audio": combined_chunk.tolist(),
                                    "sample_rate": generator.sample_rate,
                                    "chunk_num": chunk_counter,
                                    "part": f"{i+1}/{len(text_chunks)}",
                                }
                            )
                            audio_chunks = []

                except queue.Empty:
                    print(f"Timeout waiting for audio chunk {chunk_counter}", flush=True)
                    break

            # Send any remaining audio chunks
            if audio_chunks:
                combined_chunk = np.concatenate(audio_chunks)
                await websocket.send_json(
                    {
                        "type": "audio_chunk",
                        "audio": combined_chunk.tolist(),
                        "sample_rate": generator.sample_rate,
                        "chunk_num": chunk_counter,
                        "part": f"{i+1}/{len(text_chunks)}",
                    }
                )

            processing_time = time.time() - start_time
            print(f"Chunk {i+1} processed in {processing_time:.2f}s", flush=True)

        await websocket.send_json(
            {"type": "status", "message": "Audio generation complete"}
        )

    except Exception as e:
        print(f"[Audio Generation] Error: {e}", flush=True)
        await websocket.send_json(
            {"type": "error", "message": f"Generation failed: {str(e)}"}
        )

    finally:
        is_speaking = False
        audio_gen_lock.release()
        await websocket.send_json({"type": "audio_status", "status": "complete"})


def preprocess_text_for_tts(text):
    # Keep most punctuation for better speech rhythm
    pattern = r'[\\\|\/\<\>\[\]\{\}\^\`\~]'
    cleaned_text = re.sub(pattern, "", text)
    
    # Ensure proper spacing around punctuation
    cleaned_text = re.sub(r'([.,!?;:])(\S)', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'(\S)([.,!?;:])', r'\1\2', cleaned_text)
    
    # Normalize spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()


def split_text_into_meaningful_chunks(text, max_words=50):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # If no natural breaks found, split by words
    if not chunks:
        words = text.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
    
    return chunks


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

    print("[WebSocket] Client connected", flush=True)

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

                # Start audio generation
                await audio_generation(text, websocket)

    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

        # Clear queues on disconnect
        print("[Websocket] Clearing queues on disconnect...", flush=True)

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

        print("[WebSocket] Client disconnected - queues cleared", flush=True)


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
        timeout_keep_alive=300,
        timeout_graceful_shutdown=60,
    )
