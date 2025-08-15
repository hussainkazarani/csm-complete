import threading
import time
import json
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----- Config Model -----
class Config(BaseModel):
	model_path: str
	voice_speaker_id: int = 0
	reference_audio: Optional[str] = None
	reference_text: Optional[str] = None

# ----- Worker Thread -----
def model_worker():
	print("[Worker] Starting model loading...")
	time.sleep(10)
	print("[Worker] Model loading complete!")
	model_ready.set()

# ----- Globals -----
config: Config = None
model_ready = threading.Event() # will be set to True when models finish loading

# ----- FastAPI App -----
app = FastAPI()

# ----- Server Startup -----
@app.on_event("startup")
async def startupEvent():
	global config
	with open("config.json","r",encoding="utf-8") as f:
		data = json.load(f)
	config = Config(**data) # validate the data and save
	print("[Startup] Loaded config:\n", json.dumps(config.dict(), indent=4))

	# create thread
	threading.Thread(target=model_worker, daemon=True).start()
	print("[Startup] Model worker thread started.")

@app.get("/")
def root():
	return {"message": "Server is running!"}

# ----- WebSocket Connection -----
@app.websocket("/ws")
async def websocketEndpoint(websocket: WebSocket):
	await websocket.accept()
	print("[WebSocket] Client connected.")

	if not model_ready.is_set():
		await websocket.send_text("Models are loading, please wait...")
		model_ready.wait()

	await websocket.send_text("Models are ready! You can start streaming.")

	try:
		while True:
			data = await websocket.receive_text()
			print(f"[CLIENT] Received: {data.strip()}") # remove the new line of data
			await websocket.send_text(f"Echo: {data}")
	except Exception as e:
		print("[WebSocket] Error:\n", e)
	finally:
		print("[WebSocket] Client disconnected.")

#server intialization
if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)
