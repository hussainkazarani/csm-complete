import os
os.environ.update({
    "OMP_NUM_THREADS": "1",  # No CPU thread contention
    "MKL_NUM_THREADS": "1",  # No Math Kernel Library overhead
    "CUDA_LAUNCH_BLOCKING": "1",  
    "PYTORCH_DISABLE_CUDA_GRAPHS": "1",  
})

import threading
import time
import json
import torch
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional
from generator import load_csm_1b_local
import warnings

warnings.filterwarnings("ignore")

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

# ----- Worker Thread -----
def model_worker():
    # CSM Model Loading/Warm-up
    global csm_model, reference_segment

    print("[Worker] Starting model loading...", flush=True)

    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.fx_graph_cache = False
    torch.backends.cuda.matmul.allow_tf32 = True

    csm_model = load_csm_1b_local(config.model_path, "cuda")
    
    print("[Worker] CSM Model loaded. Starting warm-up...")

    for _ in csm_model.generate_stream(
        text="warm-up " * 500,
        speaker=config.voice_speaker_id,
        context=None,
        max_audio_length_ms=15000,
        temperature=0.7,
        topk=40
    ):
        pass

    print("[Worker] Model loading and Warm-up complete!", flush=True)
    model_ready.set()

# ----- Globals and App -----
app = FastAPI()
config: Config = None
model_ready = threading.Event() # will be set to True when models finish loading
csm_model = None # will store the csm model
reference_segment = None

# ----- Server Startup -----
@app.on_event("startup")
async def startupEvent():
    print("[Startup] Environment configs:")
    verify_environment()

    global config
    with open("config.json","r",encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data) # validate the data and save
    print("[Startup] Loaded config:\n", json.dumps(config.dict(), indent=4), flush=True)

    # create thread
    threading.Thread(target=model_worker, daemon=True).start()
    print("[Startup] Model worker thread started.", flush=True)

# ----- Root Page -----
@app.get("/")
def root():
    return {"message": "Server is running!"}

# ----- WebSocket Connection -----
@app.websocket("/ws")
async def websocketEndpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected.", flush=True)

    if not model_ready.is_set():
        await websocket.send_text("Models are loading, please wait...")
        model_ready.wait()

    await websocket.send_text("Models are ready! You can start streaming.")

    try:
        while True:
            data = await websocket.receive_json() # expect {"text": "..."}

            text = data["text"]

            for chunk in csm_model.generate_stream(
                    text=text,
                    speaker=config.voice_speaker_id,
                    context=None,
                    max_audio_length_ms=8000,
                    temperature=0.7,
                    topk=40
                ):
                print_gpu_memory()

                await websocket.send_bytes(chunk.cpu().numpy().tobytes()) # change to bytes before sending

            await websocket.send_json({"type": "eos"})

    except Exception as e:
        print("[WebSocket] Error:\n", e, flush=True)
    finally:
        print("[WebSocket] Client disconnected.", flush=True)

def verify_environment():
    print(f"[OMP] Threads: {os.getenv('OMP_NUM_THREADS')}")
    print(f"[CUDA] Blocking: {os.getenv('CUDA_LAUNCH_BLOCKING')}")
    print(f"[PyTorch] CUDA Graphs: {torch._inductor.config.triton.cudagraphs}")
    print(f"[TF32] MatMul: {torch.backends.cuda.matmul.allow_tf32}")

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

#server intialization
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
